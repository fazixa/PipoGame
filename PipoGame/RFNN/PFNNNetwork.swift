import Accelerate
import Foundation

/// Phase-Functioned Neural Network forward pass, ported 1:1 from the
/// `PFNN` struct in `sreyafrancis/PFNN`'s `demo/pfnn.cpp` (MODE_CUBIC only —
/// the mode the original demo's shipped video used). Two hidden layers,
/// ELU activations; the weights themselves are a cubic (Catmull-Rom)
/// interpolation across 4 phase-indexed control points out of the 50 the
/// network was trained with, re-derived every call from the live phase.
///
/// Weight data comes from a single bundled `PFNNWeights.bin`, a flat
/// concatenation (built offline, see scratchpad/pfnn_assets/) of:
/// Xmean, Xstd, Ymean, Ystd, then 4x (W0, W1, W2, b0, b1, b2) for phase
/// control points 0, 12, 25, 37 of 50 (`int(i*12.5)` for i=0...3, matching
/// pfnn.cpp's `MODE_CUBIC` loader).
final class PFNNNetwork {
    static let xDim = 342
    static let hDim = 512
    static let yDim = 311

    private let xMean: [Float]
    private let xStd: [Float]
    // Not private: pfnn.cpp's own `reset()` seeds Character's initial
    // joint state from `Ymean` (the network's trained average output —
    // a neutral standing pose) rather than zeros, and PFNNController needs
    // to replicate that same seeding.
    let yMean: [Float]
    private let yStd: [Float]

    // 4 phase-indexed weight sets, row-major (rows x cols), matching the
    // original file storage order exactly.
    private let w0: [[Float]] // each hDim * xDim
    private let w1: [[Float]] // each hDim * hDim
    private let w2: [[Float]] // each yDim * hDim
    private let b0: [[Float]] // each hDim
    private let b1: [[Float]] // each hDim
    private let b2: [[Float]] // each yDim

    init?() {
        guard let url = Bundle.main.url(forResource: "PFNNWeights", withExtension: "bin"),
              let data = try? Data(contentsOf: url) else {
            return nil
        }

        var offset = 0
        func readFloats(_ count: Int) -> [Float] {
            var result = [Float](repeating: 0, count: count)
            let byteCount = count * MemoryLayout<Float>.size
            _ = result.withUnsafeMutableBytes { dest in
                data.copyBytes(to: dest, from: offset..<(offset + byteCount))
            }
            offset += byteCount
            return result
        }

        xMean = readFloats(Self.xDim)
        xStd = readFloats(Self.xDim)
        yMean = readFloats(Self.yDim)
        yStd = readFloats(Self.yDim)

        var w0s: [[Float]] = [], w1s: [[Float]] = [], w2s: [[Float]] = []
        var b0s: [[Float]] = [], b1s: [[Float]] = [], b2s: [[Float]] = []
        for _ in 0..<4 {
            w0s.append(readFloats(Self.hDim * Self.xDim))
            w1s.append(readFloats(Self.hDim * Self.hDim))
            w2s.append(readFloats(Self.yDim * Self.hDim))
            b0s.append(readFloats(Self.hDim))
            b1s.append(readFloats(Self.hDim))
            b2s.append(readFloats(Self.yDim))
        }
        w0 = w0s; w1 = w1s; w2 = w2s
        b0 = b0s; b1 = b1s; b2 = b2s

        guard offset == data.count else {
            assertionFailure("PFNNWeights.bin size mismatch — expected to consume \(offset) bytes, file has \(data.count)")
            return nil
        }
    }

    /// `phase` is in radians, 0...2π (matches `Character::phase` in
    /// pfnn.cpp — a full gait cycle is one full turn).
    func predict(_ x: [Float], phase: Float) -> [Float] {
        precondition(x.count == Self.xDim)

        var xp = [Float](repeating: 0, count: Self.xDim)
        for i in 0..<Self.xDim { xp[i] = (x[i] - xMean[i]) / xStd[i] }

        // Cubic phase interpolation across the 4 stored control points:
        // the phase cycle (0...2π) is divided into 4 equal arcs, and the
        // surrounding 4 control points (wrapping) are Catmull-Rom blended
        // — exactly `PFNN::predict`'s MODE_CUBIC branch in pfnn.cpp.
        let cycle = (phase / (2 * .pi)) * 4
        let i1 = ((Int(cycle) % 4) + 4) % 4
        let i0 = (i1 + 3) % 4
        let i2 = (i1 + 1) % 4
        let i3 = (i1 + 2) % 4
        let mu = cycle - floor(cycle)

        let w0p = Self.cubic(w0[i0], w0[i1], w0[i2], w0[i3], mu)
        let w1p = Self.cubic(w1[i0], w1[i1], w1[i2], w1[i3], mu)
        let w2p = Self.cubic(w2[i0], w2[i1], w2[i2], w2[i3], mu)
        let b0p = Self.cubic(b0[i0], b0[i1], b0[i2], b0[i3], mu)
        let b1p = Self.cubic(b1[i0], b1[i1], b1[i2], b1[i3], mu)
        let b2p = Self.cubic(b2[i0], b2[i1], b2[i2], b2[i3], mu)

        var h0 = Self.matMulAddBias(w0p, rows: Self.hDim, cols: Self.xDim, x: xp, bias: b0p)
        Self.elu(&h0)
        var h1 = Self.matMulAddBias(w1p, rows: Self.hDim, cols: Self.hDim, x: h0, bias: b1p)
        Self.elu(&h1)
        var yp = Self.matMulAddBias(w2p, rows: Self.yDim, cols: Self.hDim, x: h1, bias: b2p)

        for i in 0..<Self.yDim { yp[i] = yp[i] * yStd[i] + yMean[i] }
        return yp
    }

    private static func cubic(_ y0: [Float], _ y1: [Float], _ y2: [Float], _ y3: [Float], _ mu: Float) -> [Float] {
        let mu2 = mu * mu, mu3 = mu2 * mu
        var c0 = -0.5 * mu3 + mu2 - 0.5 * mu
        var c1 = 1.5 * mu3 - 2.5 * mu2 + 1
        var c2 = -1.5 * mu3 + 2 * mu2 + 0.5 * mu
        var c3 = 0.5 * mu3 - 0.5 * mu2

        let n = vDSP_Length(y0.count)
        var out = [Float](repeating: 0, count: y0.count)
        var term = [Float](repeating: 0, count: y0.count)

        vDSP_vsmul(y0, 1, &c0, &out, 1, n)
        vDSP_vsmul(y1, 1, &c1, &term, 1, n)
        vDSP_vadd(out, 1, term, 1, &out, 1, n)
        vDSP_vsmul(y2, 1, &c2, &term, 1, n)
        vDSP_vadd(out, 1, term, 1, &out, 1, n)
        vDSP_vsmul(y3, 1, &c3, &term, 1, n)
        vDSP_vadd(out, 1, term, 1, &out, 1, n)
        return out
    }

    private static func matMulAddBias(_ w: [Float], rows: Int, cols: Int, x: [Float], bias: [Float]) -> [Float] {
        var result = [Float](repeating: 0, count: rows)
        vDSP_mmul(w, 1, x, 1, &result, 1, vDSP_Length(rows), 1, vDSP_Length(cols))
        vDSP_vadd(result, 1, bias, 1, &result, 1, vDSP_Length(rows))
        return result
    }

    private static func elu(_ x: inout [Float]) {
        for i in 0..<x.count where x[i] < 0 {
            x[i] = expf(x[i]) - 1
        }
    }
}
