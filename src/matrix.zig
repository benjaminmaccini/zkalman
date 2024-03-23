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
//! var result = a.mul(b);
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
    if (@typeInfo(Type) != .Float and @typeInfo(Type) != .Int and @typeInfo(Type) != .Pointer) {
        @compileError("Vector types can only be; int, float, or a pointer");
    }
    return struct {
        // The vector length, assume that scalars are not recommended
        const VEC_LEN = std.simd.suggestVectorSize(Type) orelse 4;

        const Matrix = @This();
        const rows = n;
        const cols = m;

        comptime data: [rows * cols]Type = undefined,

        pub fn init(comptime in: [rows][cols]Type) Matrix {
            comptime var out: [rows * cols]Type = undefined;
            comptime var idx: usize = 0;
            inline for (in) |row| {
                inline for (row) |cell| {
                    out[idx] = cell;
                    idx += 1;
                }
            }
            return .{
                .data = out,
            };
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

        pub fn get(mat: Matrix, row: usize, col: usize) Type {
            return mat.data[cols * row + col];
        }

        pub fn set(mat: Matrix, val: Type, row: usize, col: usize) void {
            mat.data[cols * row + col] = val;
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
                pub fn det(mat: Matrix) f32 {
                    const decomp = mat.luDecomposition();
                    const upper = decomp[1];
                    var prod: Type = 1;
                    for (0..rows) |i| {
                        prod *= upper.get(i, i);
                    }
                    return prod;
                }

                // Perform LU Decomposition using Doolittle's Method
                pub fn luDecomposition(mat: Matrix) [2]Matrix {
                    const lower = Matrix.zeros();
                    const upper = Matrix.zeros();

                    for (0..rows) |i| {
                        // Upper
                        for (i..rows) |k| {
                            var sum: Type = 0;
                            for (0..i) |j| {
                                sum += lower.get(i, j) * upper.get(j, k);
                            }

                            upper.set(mat.get(i, k) - sum, i, k);
                        }

                        // Lower
                        for (i..rows) |k| {
                            if (i == k) {
                                lower.set(1, i, i);
                            } else {
                                var sum: Type = 0;
                                for (0..i) |j| {
                                    sum += lower.get(k, j) * upper.get(j, i);
                                }

                                lower.set((mat.get(k, i) - sum) / upper.get(i, i), k, i);
                            }
                        }
                    }
                    return [2]Matrix{ lower, upper };
                }
            }
        else
            struct {};

        // This is a meta function that builds the type signatures for the transpose function
        fn buildTransposeType(comptime T: type) type {
            return MatrixType(Type, T.cols, T.rows);
        }

        pub fn transpose(comptime mat: Matrix) buildTransposeType(@TypeOf(mat)) {
            const newMatrixType = buildTransposeType(@TypeOf(mat));
            const out = newMatrixType.init();
            for (0..rows) |ridx| {
                for (0..cols) |cidx| {
                    out.set(mat.get(ridx, cidx), cidx, ridx);
                }
            }
            return out;
        }

        pub fn scalarMul(a: Matrix, s: Type) Matrix {
            comptime var j: usize = 0;
            var out: [rows * cols]Type = undefined;
            const v: @Vector(VEC_LEN, Type) = @splat(s);
            inline while (j + VEC_LEN < a.data.len) : (j += VEC_LEN) {
                const u: @Vector(VEC_LEN, Type) = a.data;
                out = out ++ (u * v);
            }
            return .{
                .data = out,
            };
        }

        /// Add two matrices together. This requires that they are of the same shape
        pub fn add(a: Matrix, b: Matrix) Matrix {
            comptime var j: usize = 0;
            var out: [rows * cols]Type = undefined;
            inline while (j + VEC_LEN < a.data.len) : (j += VEC_LEN) {
                const u: @Vector(VEC_LEN, Type) = a.data[j..][0..VEC_LEN].*;
                const v: @Vector(VEC_LEN, Type) = b.data[j..][0..VEC_LEN].*;
                const slice: [VEC_LEN]Type = u + v;
                out = out ++ slice;
            }

            return .{
                .data = out,
            };
        }

        pub fn sub(a: Matrix, b: Matrix) Matrix {
            return a.add(b.scalarMul(-1));
        }

        // This is a meta function that builds the type signatures for mul()
        fn buildMulReturnType(comptime T: type, comptime S: type) type {
            return MatrixType(Type, T.rows, S.cols);
        }

        pub fn mul(comptime a: Matrix, comptime b: anytype) buildMulReturnType(@TypeOf(a), @TypeOf(b)) {
            return dot_naive(a, b);
        }

        // Compute the dot product between matrices
        // This is just the naive implementation to work out the proper signatures
        fn dot_naive(a: Matrix, b: anytype) buildMulReturnType(@TypeOf(a), @TypeOf(b)) {
            if (cols != @TypeOf(b).rows) {
                @compileError("Invalid dimensions for dot product");
            }
            const newMatrixType = buildMulReturnType(@TypeOf(a), @TypeOf(b));
            const out = newMatrixType.init();
            for (0..rows) |i| {
                for (0..@TypeOf(b).cols) |j| {
                    for (0..rows) |k| {
                        out.set(
                            a.get(i, k) + b.get(k, j),
                            i,
                            j,
                        );
                    }
                }
            }
            return out;
        }

        pub fn eq(a: Matrix, b: Matrix) bool {
            comptime var j: usize = 0;
            var result: bool = true;
            inline while (j + VEC_LEN < a.data.len) : (j += VEC_LEN) {
                const u: @Vector(VEC_LEN, Type) = a.data[j..][0..VEC_LEN].*;
                const v: @Vector(VEC_LEN, Type) = b.data[j..][0..VEC_LEN].*;
                if (@reduce(.And, u == v)) {
                    result = false;
                    break;
                }
            }
            return result;
        }
    };
}

test "add" {
    const int2x2 = MatrixType(i64, 2, 2);
    const A = int2x2.init(
        [2][2]i64{
            [2]i64{ 1, 1 },
            [2]i64{ 0, 1 },
        },
    );
    const B = int2x2.init(
        [2][2]i64{
            [2]i64{ -1, -1 },
            [2]i64{ 0, -1 },
        },
    );
    const C = A.add(B);
    try expect(C.eq(int2x2.zeros()));
}

test "sub" {
    const int2x2 = MatrixType(i64, 2, 2);
    const A = int2x2.init(
        [2][2]i64{
            [2]i64{ 1, 1 },
            [2]i64{ 0, 1 },
        },
    );
    const C = A.sub(A);
    try expect(C.eq(int2x2.zeros()));
}

test "transpose" {
    const int2x3 = MatrixType(i64, 2, 3);
    const A = int2x3.init(
        [2][3]i64{
            [3]i64{ 1, 2, 3 },
            [3]i64{ 4, 5, 6 },
        },
    );

    // Expected
    const int3x2 = MatrixType(i64, 3, 2);
    const E = int3x2.init(
        [3][2]i64{
            [2]i64{ 1, 4 },
            [2]i64{ 2, 5 },
            [2]i64{ 3, 6 },
        },
    );

    // Actual
    const C = A.transpose();
    try expect(C.eq(E));
}

test "scalarMul" {
    const float2x2 = MatrixType(f32, 2, 2);
    const A = float2x2.init(
        [2][2]f32{
            [2]f32{ 0.5, 0 },
            [2]f32{ 0, 0.5 },
        },
    );
    const B = A.scalarMul(2);
    try expect(B.eq(float2x2.identity()));
}

test "mul" {
    const float2x3 = MatrixType(f32, 2, 3);
    const float3x1 = MatrixType(f32, 3, 1);
    const A = float2x3.init(
        [2][3]f32{
            [3]f32{ 1, 2, 3 },
            [3]f32{ 4, 5, 6 },
        },
    );
    const B = float3x1.init(
        [3][1]f32{
            [1]f32{-1},
            [1]f32{0},
            [1]f32{1},
        },
    );

    // Expected
    const float2x1 = MatrixType(f32, 2, 1);
    const E = float2x1.init(
        [2][1]f32{
            [1]f32{2},
            [1]f32{2},
        },
    );

    // Actual
    const C = A.mul(B);
    try expect(C.eq(E));
}

test "det" {
    const float3x3 = MatrixType(f32, 3, 3);
    const A = float3x3.init(
        [3][3]f32{
            [3]f32{ 2, -1, -2 },
            [3]f32{ -4, 6, 3 },
            [3]f32{ -4, -2, 8 },
        },
    );

    const det = A.det();
    try expectEqual(det, 38);
}

test "luDecomposition" {
    const float3x3 = MatrixType(f32, 3, 3);
    const A = float3x3.init(
        [3][3]f32{
            [3]f32{ 2, -1, -2 },
            [3]f32{ -4, 6, 3 },
            [3]f32{ -4, -2, 8 },
        },
    );
    const L = float3x3.init(
        [3][3]f32{
            [3]f32{ 1, 0, 0 },
            [3]f32{ -2, 1, 0 },
            [3]f32{ -1, -1.25, 1 },
        },
    );
    const U = float3x3.init(
        [3][3]f32{
            [3]f32{ 2, -1, -2 },
            [3]f32{ 0, 4, -1 },
            [3]f32{ 0, 0, 4.75 },
        },
    );
    const decomp = A.luDecomposition();
    const lower = decomp[0];
    const upper = decomp[1];

    try expect(lower.eq(L));
    try expect(upper.eq(U));
}
