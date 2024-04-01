//! This file contains a generic vector and martix type. It features
//! compile time checking for the matrix dimensions and SIMD-executed
//! matrix operations using the builtin Vector type, hopefully for
//! a faster and safer coding experience
//!
//! Gotchas:
//! - Large matrices may cause compiler crashes (Zig<=0.11.0)
//!
//! Basic example:
//! const std = @import("std")
//! const Mat = @import("math.zig").Mat
//!
//! const int2x2 = MatrixType(i64, 2, 2);
//! const int2x1 = MatrixType(i64, 2, 1);
//! var a = int2x2.new(
//!     [2][2]i64{
//!         [_]i64{1, 1},
//!         [_]i64{1, 0},
//!     },
//! );
//! var b = int2x1.new(
//!     [2][1]i64{
//!         [_]i64{1},
//!         [_]i64{1},
//!     },
//! );
//! var result = a.mul(&b);
//! std.debug.assert(std.mem.eql(i64, result.shape(), [2]i64{2, 1}));
//! std.debug.assert(result.data[0][0] == 2);
//! std.debug.assert(result.data[1][0] == 1);
const std = @import("std");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

/// This is a generic matrix structure. It uses static memory (e.g. no allocator)
/// and enforces dimension checking at compile time.
pub fn MatrixType(
    comptime Type: type,
    comptime n: usize,
    comptime m: usize,
) type {
    if (@typeInfo(Type) != .Bool and @typeInfo(Type) != .Float and @typeInfo(Type) != .Int and @typeInfo(Type) != .Pointer) {
        @compileError("Vector types can only be; bool, int, float, or a pointer");
    }
    return struct {
        // The vector length, assume that scalars are not recommended
        const VEC_LEN = std.simd.suggestVectorSize(Type) orelse 4;

        const Matrix = @This();
        const rows = n;
        const cols = m;

        data: [rows * cols]Type,

        pub fn init(comptime in: [rows][cols]Type) Matrix {
            var newMat = Matrix{ .data = undefined };
            inline for (in, 0..) |row, ridx| {
                inline for (row, 0..) |cell, cidx| {
                    newMat.set(cell, ridx, cidx);
                }
            }
            return newMat;
        }

        pub fn zeros() Matrix {
            var out = [_]Type{0} ** (rows * cols);
            return .{
                .data = out,
            };
        }

        pub fn shape() [2]i64 {
            return [2]i64{ rows, cols };
        }

        pub inline fn get(mat: *Matrix, row: usize, col: usize) Type {
            return mat.data[cols * row + col];
        }

        pub inline fn set(mat: *Matrix, val: Type, row: usize, col: usize) void {
            mat.data[cols * row + col] = val;
        }

        // This is a meta function that builds the type signatures for the column vector
        fn buildColumnVectorType() type {
            return MatrixType(Type, rows, 1);
        }

        // This is a meta function that builds the type signatures for the row vector
        fn buildRowVectorType() type {
            return MatrixType(Type, 1, cols);
        }

        pub fn getCol(mat: *Matrix, cidx: usize) buildColumnVectorType() {
            var vecType = buildColumnVectorType();
            var vecData: [rows]Type = undefined;
            for (0..rows) |ridx| {
                vecData.set(mat.get(ridx, cidx), ridx, cidx);
            }
            return vecType{
                .data = vecData,
            };
        }

        pub fn setRow(mat: *Matrix, ridx: usize, row: buildRowVectorType()) Matrix {
            for (0..cols) |cidx| {
                mat.set(row.get(0, cidx), ridx, cidx);
            }
            return mat;
        }

        pub fn setCol(mat: *Matrix, cidx: usize, col: buildColumnVectorType()) Matrix {
            for (0..rows) |ridx| {
                mat.set(col.get(ridx, 0), ridx, cidx);
            }
            return mat;
        }

        pub fn swapRows(mat: *Matrix, ridx1: usize, ridx2: usize) Matrix {
            for (mat.data[cols * ridx1 .. (ridx1 + 1) * cols], mat.data[cols * ridx2 .. (ridx2 + 1) * cols], 0..) |r1, r2, i| {
                const tempVal = r1;
                mat.set(r2, ridx1, i);
                mat.set(tempVal, ridx2, i);
            }
            return mat.*;
        }

        // Only expose certain properties for square matrices
        pub usingnamespace if (rows == cols)
            struct {
                pub fn identity() Matrix {
                    var out: [rows * cols]Type = undefined;
                    for (0..(rows * cols)) |idx| {
                        out[idx] = if (idx % (rows + 1) == 0) 1 else 0;
                    }
                    return .{ .data = out };
                }

                // Matrix computes the determinant for the matrix data
                // using the LU decomposition algorithm
                pub fn det(mat: *Matrix) Type {
                    var p = mat.pivot();
                    var decomp = mat.luDecomposition(&p[0]);
                    // Get the pivot's determinant
                    var det_p = std.math.pow(Type, -1, p[1]);

                    // Get the determinant for the upper matrix
                    var upper = decomp[2];
                    var prod: Type = 1;
                    for (0..rows) |i| {
                        prod *= upper.get(i, i);
                    }
                    return det_p * prod;
                }

                pub fn inv(mat: *Matrix) !Matrix {
                    // LU Decomposition yields LU = PA
                    // this is the same as LUA^-1 = P
                    var plu = mat.luDecomposition(null);
                    var P = plu[0];
                    var L = plu[1];
                    var U = plu[2];

                    // Check singular
                    var prod: Type = 1;
                    for (0..rows) |i| {
                        prod *= U.get(i, i);
                    }
                    const eqStatement = switch (@typeInfo(Type)) {
                        .Float => @fabs(prod) < 1e-6,
                        .Int => prod == 0,
                        else => unreachable,
                    };
                    if (eqStatement) {
                        return error.SingularMatrixHasNoInverse;
                    }

                    // For each column n of P, solve the linear system: LUa_n = p_n
                    // using forward substitution, followed by back substitution
                    // This is specific for linear equations of the form LUB = P
                    var B = Matrix.zeros();
                    for (0..cols) |cidx| {
                        for (0..rows) |i| {
                            B.set(P.get(i, cidx), i, cidx);
                            for (0..i) |j| {
                                B.set(B.get(i, cidx) - L.get(i, j) * B.get(j, cidx), i, cidx);
                            }
                            B.set(B.get(i, cidx) / L.get(i, i), i, cidx);
                        }
                    }
                    for (0..cols) |cidx| {
                        var i: usize = rows - 1;
                        while (i > 0) : (i -= 1) {
                            for (i + 1..rows) |j| {
                                B.set(B.get(i, cidx) - U.get(i, j) * B.get(j, cidx), i, cidx);
                            }
                            B.set(B.get(i, cidx) / U.get(i, i), i, cidx);
                        } else {
                            // There might be a better way to do this, but Zig requires a positive
                            // integer for loops, but we are iterating backwards in this case. To
                            // prevent an overflow error, we can just handle the final case here instead
                            for (i + 1..rows) |j| {
                                B.set(B.get(i, cidx) - U.get(i, j) * B.get(j, cidx), i, cidx);
                            }
                            B.set(B.get(i, cidx) / U.get(i, i), i, cidx);
                        }
                    }
                    return B;
                }

                // Return the pivot matrix and the number of swaps
                // For each column, find the max value below (and including) the current diagonal
                pub fn pivot(mat: *Matrix) struct { Matrix, Type } {
                    var p = Matrix.identity();
                    var numSwaps: Type = 0;
                    for (0..cols) |cidx| {
                        var max: Type = undefined;
                        var midx: usize = 0;
                        for (cidx..rows) |ridx| {
                            const newMax = @max(mat.get(ridx, cidx), max);
                            if (newMax != max) {
                                max = newMax;
                                midx = cidx;
                            }
                        }

                        // Swap the rows
                        if (midx != cidx) {
                            numSwaps += 1;
                            p = p.swapRows(midx, cidx);
                        }
                    }
                    return .{ p, numSwaps };
                }

                // Perform LU Decomposition with partial pivoting to avoid rounding errors
                // Optionally, provide a pivot matrix. Useful for when computing the determinant
                // when the number of swaps needs to be kept track of
                pub fn luDecomposition(mat: *Matrix, p: ?*Matrix) [3]Matrix {
                    var I = Matrix.identity();
                    if (mat.eq(&I)) {
                        return [3]Matrix{ I, I, I };
                    }
                    var P = @constCast(p orelse &mat.pivot()[0]).*;
                    var L = Matrix.zeros();
                    var U = Matrix.zeros();
                    var PA = P.mul(mat);

                    for (0..rows) |j| {
                        // Set lower to unity
                        L.set(1, j, j);

                        // Upper
                        for (0..j + 1) |i| {
                            var sum: Type = 0;
                            for (0..i) |k| {
                                sum += L.get(i, k) * U.get(k, j);
                            }
                            U.set(PA.get(i, j) - sum, i, j);
                        }

                        // Lower
                        for (j..rows) |i| {
                            var sum: Type = 0;
                            for (0..j) |k| {
                                sum += L.get(i, k) * U.get(k, j);
                            }

                            L.set((PA.get(i, j) - sum) / U.get(j, j), i, j);
                        }
                    }
                    return [3]Matrix{ P, L, U };
                }
            }
        else
            struct {};

        // This is a meta function that builds the type signatures for the transpose function
        fn buildTransposeType() type {
            return MatrixType(Type, cols, rows);
        }

        pub fn transpose(mat: *Matrix) buildTransposeType() {
            const newMatrixType = buildTransposeType();
            var out = newMatrixType{ .data = undefined };
            for (0..rows) |ridx| {
                for (0..cols) |cidx| {
                    out.set(mat.get(ridx, cidx), cidx, ridx);
                }
            }
            return out;
        }

        pub fn scalarMul(a: *Matrix, s: Type) Matrix {
            comptime var j: usize = 0;
            var out: [rows * cols]Type = undefined;
            const v: @Vector(VEC_LEN, Type) = @splat(s);
            inline while (j + VEC_LEN < a.data.len) : (j += VEC_LEN) {
                const u: @Vector(VEC_LEN, Type) = a.data[j..][0..VEC_LEN].*;
                const slice: [VEC_LEN]Type = u * v;
                @memcpy(out[j .. j + VEC_LEN], &slice);
            }
            return .{
                .data = out,
            };
        }

        /// Add two matrices together. This requires that they are of the same shape
        pub fn add(a: *Matrix, b: *Matrix) Matrix {
            comptime var j: usize = 0;
            var out: [rows * cols]Type = undefined;
            inline while (j + VEC_LEN < a.data.len) : (j += VEC_LEN) {
                const u: @Vector(VEC_LEN, Type) = a.data[j..][0..VEC_LEN].*;
                const v: @Vector(VEC_LEN, Type) = b.data[j..][0..VEC_LEN].*;
                const slice: [VEC_LEN]Type = u + v;
                @memcpy(out[j .. j + VEC_LEN], &slice);
            }

            return .{
                .data = out,
            };
        }

        pub fn sub(a: *Matrix, b: *Matrix) Matrix {
            var neg = b.scalarMul(-1);
            return a.add(&neg);
        }

        // This is a meta function that builds the type signatures for mul()
        fn buildMulReturnType(comptime T: type, comptime S: type) type {
            return MatrixType(Type, T.rows, S.cols);
        }

        pub fn mul(a: *Matrix, b: anytype) buildMulReturnType(@TypeOf(a.*), @TypeOf(b.*)) {
            return dot_naive(a, b);
        }

        // Compute the dot product between matrices
        // This is just the naive implementation to work out the proper signatures
        fn dot_naive(a: *Matrix, b: anytype) buildMulReturnType(@TypeOf(a.*), @TypeOf(b.*)) {
            if (cols != @TypeOf(b.*).rows) {
                @compileError("Invalid dimensions for dot product");
            }
            const newMatrixType = buildMulReturnType(@TypeOf(a.*), @TypeOf(b.*));
            var out = newMatrixType{ .data = undefined };
            for (0..rows) |i| {
                for (0..@TypeOf(b.*).cols) |j| {
                    var v: Type = 0;
                    for (0..rows) |k| {
                        v += a.get(i, k) * b.get(k, j);
                    }
                    out.set(v, i, j);
                }
            }
            return out;
        }

        pub fn eq(a: *Matrix, b: *Matrix) bool {
            comptime var j: usize = 0;
            var result: bool = true;
            inline while (j + VEC_LEN < a.data.len) : (j += VEC_LEN) {
                const u: @Vector(VEC_LEN, Type) = a.data[j..][0..VEC_LEN].*;
                const v: @Vector(VEC_LEN, Type) = b.data[j..][0..VEC_LEN].*;

                const eqStatement = switch (@typeInfo(Type)) {
                    inline .Bool => u == v,
                    inline .Float => blk: {
                        const close: @Vector(VEC_LEN, Type) = @splat(1e-6);
                        break :blk @fabs(u - v) < close;
                    },
                    inline .Int => u == v,
                    else => unreachable,
                };
                if (!@reduce(.And, eqStatement)) {
                    // std.debug.print("\n{} != {}\n", .{ a, b });
                    result = false;
                    break;
                }
            }
            return result;
        }

        pub fn format(
            mat: Matrix,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            var matPtr: *Matrix = @constCast(&mat);
            _ = fmt;
            _ = options;
            try writer.writeAll(@typeName(Matrix));
            try writer.writeAll("[\n");
            for (0..rows) |ridx| {
                try writer.writeAll("\t[");
                for (0..cols) |cidx| {
                    try writer.print("{d:.4}", .{matPtr.get(ridx, cidx)});
                    if (cidx != cols - 1) {
                        try writer.writeByte(',');
                    }
                }
                try writer.writeByte(']');
                if (ridx != rows - 1) {
                    try writer.writeAll(",\n");
                }
            }
            try writer.writeAll("]\n");
        }
    };
}

test "add" {
    const int2x2 = MatrixType(i64, 2, 2);
    var A = int2x2.init(
        [2][2]i64{
            [2]i64{ 1, 1 },
            [2]i64{ 0, 1 },
        },
    );
    var B = int2x2.init(
        [2][2]i64{
            [2]i64{ -1, -1 },
            [2]i64{ 0, -1 },
        },
    );
    var C = A.add(&B);
    var zeros = int2x2.zeros();
    try expect(C.eq(&zeros));
}

test "sub" {
    const int2x2 = MatrixType(i64, 2, 2);
    var A = int2x2.init(
        [2][2]i64{
            [2]i64{ 1, 1 },
            [2]i64{ 0, 1 },
        },
    );
    var C = A.sub(&A);
    var zeros = int2x2.zeros();
    try expect(C.eq(&zeros));
}

test "swapRows" {
    const int3x3 = MatrixType(i64, 3, 3);
    var A = int3x3.init(
        [3][3]i64{
            [3]i64{ 1, 2, 3 },
            [3]i64{ 0, 0, 0 },
            [3]i64{ 4, 5, 6 },
        },
    );
    var E = int3x3.init(
        [3][3]i64{
            [3]i64{ 4, 5, 6 },
            [3]i64{ 0, 0, 0 },
            [3]i64{ 1, 2, 3 },
        },
    );

    A = A.swapRows(0, 2);
    try expect(A.eq(&E));
}

test "pivot" {
    const float3x3 = MatrixType(f32, 3, 3);
    var A = float3x3.init(
        [3][3]f32{
            [3]f32{ 1, 2, 3 },
            [3]f32{ 2, -4, 6 },
            [3]f32{ 3, -9, -3 },
        },
    );
    var E = float3x3.init(
        [3][3]f32{
            [3]f32{ 0, 0, 1 },
            [3]f32{ 1, 0, 0 },
            [3]f32{ 0, 1, 0 },
        },
    );

    var p = A.pivot();

    try expect(p[0].eq(&E));
    try expectEqual(p[1], 2);
}

test "transpose" {
    const int2x3 = MatrixType(i64, 2, 3);
    var A = int2x3.init(
        [2][3]i64{
            [3]i64{ 1, 2, 3 },
            [3]i64{ 4, 5, 6 },
        },
    );

    // Expected
    const int3x2 = MatrixType(i64, 3, 2);
    var E = int3x2.init(
        [3][2]i64{
            [2]i64{ 1, 4 },
            [2]i64{ 2, 5 },
            [2]i64{ 3, 6 },
        },
    );

    // Actual
    var C = A.transpose();
    try expect(C.eq(&E));
}

test "scalarMul" {
    const float2x2 = MatrixType(f32, 2, 2);
    var A = float2x2.init(
        [2][2]f32{
            [2]f32{ 0.5, 0 },
            [2]f32{ 0, 0.5 },
        },
    );
    var B = A.scalarMul(2);
    var id = float2x2.identity();
    try expect(B.eq(&id));
}

test "mul" {
    const float2x3 = MatrixType(f32, 2, 3);
    const float3x1 = MatrixType(f32, 3, 1);
    var A = float2x3.init(
        [2][3]f32{
            [3]f32{ 1, 2, 3 },
            [3]f32{ 4, 5, 6 },
        },
    );
    var B = float3x1.init(
        [3][1]f32{
            [1]f32{-1},
            [1]f32{0},
            [1]f32{1},
        },
    );

    // Expected
    const float2x1 = MatrixType(f32, 2, 1);
    var E = float2x1.init(
        [2][1]f32{
            [1]f32{2},
            [1]f32{2},
        },
    );

    // Actual
    var C = A.mul(&B);
    try expect(C.eq(&E));
}

test "mul2" {
    const float3x3 = MatrixType(f32, 3, 3);
    var A = float3x3.init(
        [3][3]f32{
            [3]f32{ 1, 2, 3 },
            [3]f32{ 4, 5, 6 },
            [3]f32{ 7, 8, 9 },
        },
    );
    var B = float3x3.init(
        [3][3]f32{
            [3]f32{ 1, -1, 0 },
            [3]f32{ -1, 0, 1 },
            [3]f32{ 0, -1, 1 },
        },
    );

    // Expected
    var E = float3x3.init(
        [3][3]f32{
            [3]f32{ -1, -4, 5 },
            [3]f32{ -1, -10, 11 },
            [3]f32{ -1, -16, 17 },
        },
    );

    // Actual
    var C = A.mul(&B);
    try expect(C.eq(&E));
}
test "det" {
    const float3x3 = MatrixType(f32, 3, 3);
    var A = float3x3.init(
        [3][3]f32{
            [3]f32{ 2, -1, -2 },
            [3]f32{ -4, 6, 3 },
            [3]f32{ -4, -2, 8 },
        },
    );

    const det = A.det();
    try expectEqual(det, 24);
}

test "luDecomposition" {
    const float3x3 = MatrixType(f32, 3, 3);
    var A = float3x3.init(
        [3][3]f32{
            [3]f32{ 1, 2, 3 },
            [3]f32{ 2, -4, 6 },
            [3]f32{ 3, -9, -3 },
        },
    );
    var L = float3x3.init(
        [3][3]f32{
            [3]f32{ 1, 0, 0 },
            [3]f32{ 1.0 / 3.0, 1, 0 },
            [3]f32{ 2.0 / 3.0, 2.0 / 5.0, 1 },
        },
    );
    var U = float3x3.init(
        [3][3]f32{
            [3]f32{ 3, -9, -3 },
            [3]f32{ 0, 5, 4 },
            [3]f32{ 0, 0, 32.0 / 5.0 },
        },
    );
    var P = float3x3.init(
        [3][3]f32{
            [3]f32{ 0, 0, 1 },
            [3]f32{ 1, 0, 0 },
            [3]f32{ 0, 1, 0 },
        },
    );
    var decomp = A.luDecomposition(null);

    // Check PA = LU
    var PA = decomp[0].mul(&A);
    var LU = decomp[1].mul(&decomp[2]);

    try expect(PA.eq(&LU));

    try expect(decomp[0].eq(&P));
    try expect(decomp[1].eq(&L));
    try expect(decomp[2].eq(&U));
}

test "inv" {
    const int3x3 = MatrixType(u8, 3, 3);
    var I = int3x3.init([3][3]u8{
        [3]u8{ 1, 0, 0 },
        [3]u8{ 0, 1, 0 },
        [3]u8{ 0, 0, 1 },
    });

    var I_inv = try I.inv();
    try expect(I.eq(&I_inv));
}

test "inv2" {
    const float3x3 = MatrixType(f32, 3, 3);
    var A = float3x3.init(
        [3][3]f32{
            [3]f32{ 1, 2, 3 },
            [3]f32{ 2, -4, 6 },
            [3]f32{ 3, -9, -3 },
        },
    );

    var Ep = float3x3.init(
        [3][3]f32{
            [3]f32{ 66, -21, 24 },
            [3]f32{ 24, -12, 0 },
            [3]f32{ -6, 15, -8 },
        },
    );
    var E = Ep.scalarMul(1.0 / 96.0);
    var B = try A.inv();
    try expect(B.eq(&E));
}

test "inv3" {
    const float2x2 = MatrixType(f32, 2, 2);
    var A = float2x2.init(
        [2][2]f32{
            [2]f32{ 2, 1 },
            [2]f32{ 1, 1 },
        },
    );
    var E = float2x2.init(
        [2][2]f32{
            [2]f32{ 1, -1 },
            [2]f32{ -1, 2 },
        },
    );

    var B = try A.inv();
    try expect(B.eq(&E));
}
