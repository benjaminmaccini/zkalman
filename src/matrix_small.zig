//! This file provides implementations for low dim matrices (<4) with the intent of
//! overriding potentially expensive operations meant for larger matrices:
//!   - Inverse
//!   - Multiplication
//!   - Determinant
const std = @import("std");
const MatrixType = @import("matrix.zig").MatrixType;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

pub fn Matrix2x2Type(comptime Type: type) type {
    return struct {
        fn VectorMixin(comptime Vector: type) type {
            return struct {
                pub fn add(a: Vector, b: Vector) Vector {
                    var newVec: Vector = undefined;
                    inline for (@typeInfo(Vector).Struct.fields) |fld| {
                        @field(newVec, fld.name) = @field(a, fld.name) + @field(b, fld.name);
                    }
                    return newVec;
                }

                pub fn sub(a: Vector, b: Vector) Vector {
                    var newVec: Vector = undefined;
                    inline for (@typeInfo(Vector).Struct.fields) |fld| {
                        @field(newVec, fld.name) = @field(a, fld.name) - @field(b, fld.name);
                    }
                    return newVec;
                }

                pub fn mul(a: Vector, b: Vector) Vector {
                    var newVec: Vector = undefined;
                    inline for (@typeInfo(Vector).Struct.fields) |fld| {
                        @field(newVec, fld.name) = @field(a, fld.name) * @field(b, fld.name);
                    }
                    return newVec;
                }

                pub fn div(a: Vector, b: Vector) Vector {
                    var newVec: Vector = undefined;
                    inline for (@typeInfo(Vector).Struct.fields) |fld| {
                        @field(newVec, fld.name) = @field(a, fld.name) / @field(b, fld.name);
                    }
                    return newVec;
                }

                pub fn eq(a: Vector, b: Vector) bool {
                    inline for (@typeInfo(Vector).Struct.fields) |fld| {
                        if (@field(a, fld.name) != @field(b, fld.name)) {
                            return false;
                        }
                    }
                    return true;
                }
            };
        }

        pub const Vec1 = extern struct {
            const Self = @This();
            pub usingnamespace VectorMixin(Self);

            x: Type,

            pub fn init(comptime in: [1]Type) Self {
                return .{ .x = in[0] };
            }
        };

        pub const Vec2 = extern struct {
            const Self = @This();
            pub usingnamespace VectorMixin(Self);

            x: Type,
            y: Type,

            pub fn init(comptime in: [2]Type) Self {
                return .{
                    .x = in[0],
                    .y = in[1],
                };
            }
        };

        pub const Vec3 = extern struct {
            const Self = @This();
            pub usingnamespace VectorMixin(Self);

            x: Type,
            y: Type,
            z: Type,

            pub fn init(comptime in: [3]Type) Self {
                return .{
                    .x = in[0],
                    .y = in[1],
                    .z = in[2],
                };
            }
        };

        pub const Mat2x2 = struct {
            const Self = @This();
            pub usingnamespace MatrixType(Self, 2, 2);

            pub fn det(mat: *Mat2x2) Type {
                _ = mat;
            }

            pub fn inv(mat: *Mat2x2) !Mat2x2 {
                _ = mat;
            }

            pub fn mulByVec2(mat: *Mat2x2, vec: *Vec2) Vec2 {
                _ = vec;
                _ = mat;
            }

            pub fn mulByMat2x3(a: *Mat2x2, b: *Mat2x3) Mat2x3 {
                _ = b;
                _ = a;
            }
        };

        pub const Mat2x3 = extern struct {
            const Self = @This();
            pub usingnamespace MatrixType(Self, 2, 3);

            pub fn transponse(mat: *Mat2x3) Mat3x2 {
                return .{ .data = [6]Type{
                    mat.get(0, 0),
                    mat.get(1, 0),
                    mat.get(0, 1),
                    mat.get(1, 1),
                    mat.get(0, 2),
                    mat.get(1, 2),
                } };
            }

            pub fn mulByVec2(mat: *Mat2x3, vec: *Vec3) Vec2 {
                return .{
                    .x = mat.get(0, 0) * vec.x + mat.get(0, 1) * vec.y + mat.get(0, 2) * vec.z,
                    .y = mat.get(1, 0) * vec.x + mat.get(1, 1) * vec.y + mat.get(1, 2) * vec.z,
                };
            }

            pub fn mulByMat3x3(a: *Mat2x3, b: *Mat3x3) Mat2x3 {
                var newMat = Mat2x3{ .data = undefined };
                newMat.set(a.get(0, 0) * b.get(0, 0) + a.get(0, 1) * b.get(1, 0) + a.get(0, 2) * b.get(2, 0), 0, 0);
                newMat.set(a.get(0, 0) * b.get(0, 1) + a.get(0, 1) * b.get(1, 1) + a.get(0, 2) * b.get(2, 1), 0, 1);
                newMat.set(a.get(0, 0) * b.get(0, 2) + a.get(0, 1) * b.get(1, 2) + a.get(0, 2) * b.get(2, 2), 0, 2);
                newMat.set(a.get(1, 0) * b.get(0, 0) + a.get(1, 1) * b.get(1, 0) + a.get(1, 2) * b.get(2, 0), 1, 0);
                newMat.set(a.get(1, 0) * b.get(0, 1) + a.get(1, 1) * b.get(1, 1) + a.get(1, 2) * b.get(2, 1), 1, 1);
                newMat.set(a.get(1, 0) * b.get(0, 2) + a.get(1, 1) * b.get(1, 2) + a.get(1, 2) * b.get(2, 2), 1, 2);
                return newMat;
            }

            pub fn mulByMat3x2(a: *Mat2x3, b: *Mat3x2) Mat2x2 {
                var newMat = Mat2x2{ .data = undefined };
                newMat.set(a.get(0, 0) * b.get(0, 0) + a.get(0, 1) * b.get(1, 0) + a.get(0, 2) * b.get(2, 0), 0, 0);
                newMat.set(a.get(0, 0) * b.get(0, 1) + a.get(0, 1) * b.get(1, 1) + a.get(0, 2) * b.get(2, 1), 0, 1);
                newMat.set(a.get(1, 0) * b.get(0, 0) + a.get(1, 1) * b.get(1, 0) + a.get(1, 2) * b.get(2, 0), 1, 0);
                newMat.set(a.get(1, 0) * b.get(0, 1) + a.get(1, 1) * b.get(1, 1) + a.get(1, 2) * b.get(2, 1), 1, 1);
                return newMat;
            }
        };

        pub const Mat3x2 = extern struct {
            const Self = @This();
            pub usingnamespace MatrixType(Self);

            pub fn transponse(mat: *Mat3x2) Mat2x3 {
                return .{ .data = [6]Type{
                    mat.get(0, 0),
                    mat.get(1, 0),
                    mat.get(2, 0),
                    mat.get(1, 0),
                    mat.get(1, 1),
                    mat.get(1, 2),
                } };
            }

            pub fn mulByVec2(mat: *Mat3x2, vec: *Vec2) Vec3 {
                return .{
                    .x = mat.get(0, 0) * vec.x + mat.get(0, 1) * vec.y,
                    .y = mat.get(1, 0) * vec.x + mat.get(1, 1) * vec.y,
                    .z = mat.get(2, 0) * vec.x + mat.get(2, 1) * vec.y,
                };
            }

            pub fn mulByMat2x2(a: *Mat3x2, b: *Mat2x2) Mat3x2 {
                var newMat = Mat3x2{ .data = undefined };
                newMat.set(a.get(0, 0) * b.get(0, 0) + a.get(0, 1) * b.get(1, 0), 0, 0);
                newMat.set(a.get(0, 0) * b.get(0, 1) + a.get(0, 1) * b.get(1, 1), 0, 1);
                newMat.set(a.get(1, 0) * b.get(0, 0) + a.get(1, 1) * b.get(1, 0), 1, 0);
                newMat.set(a.get(1, 0) * b.get(0, 1) + a.get(1, 1) * b.get(1, 1), 1, 1);
                newMat.set(a.get(2, 0) * b.get(0, 0) + a.get(2, 1) * b.get(1, 0), 2, 0);
                newMat.set(a.get(2, 0) * b.get(0, 1) + a.get(2, 1) * b.get(1, 1), 2, 1);
                return newMat;
            }

            pub fn mulByMat2x3(a: *Mat3x2, b: *Mat2x3) Mat3x3 {
                var newMat = Mat3x3{ .data = undefined };
                newMat.set(a.get(0, 0) * b.get(0, 0) + a.get(0, 1) * b.get(1, 0), 0, 0);
                newMat.set(a.get(0, 0) * b.get(0, 1) + a.get(0, 1) * b.get(1, 1), 0, 1);
                newMat.set(a.get(0, 0) * b.get(0, 2) + a.get(0, 1) * b.get(1, 2), 0, 2);
                newMat.set(a.get(1, 0) * b.get(0, 0) + a.get(1, 1) * b.get(1, 0), 1, 0);
                newMat.set(a.get(1, 0) * b.get(0, 1) + a.get(1, 1) * b.get(1, 1), 1, 1);
                newMat.set(a.get(1, 0) * b.get(0, 2) + a.get(1, 1) * b.get(1, 2), 1, 2);
                newMat.set(a.get(2, 0) * b.get(0, 0) + a.get(2, 1) * b.get(1, 0), 2, 0);
                newMat.set(a.get(2, 0) * b.get(0, 1) + a.get(2, 1) * b.get(1, 1), 2, 1);
                newMat.set(a.get(2, 0) * b.get(0, 2) + a.get(2, 1) * b.get(1, 2), 2, 2);
                return newMat;
            }
        };

        pub const Mat3x3 = extern struct {
            const Self = @This();
            pub usingnamespace MatrixType(Self);

            pub fn det(mat: *Mat3x3) Type {
                _ = mat;
            }
            pub fn inv(mat: *Mat3x3) !Mat3x3 {
                _ = mat;
            }
            pub fn mulByVec3(mat: *Mat3x3, vec: *Vec3) Vec3 {
                _ = vec;
                _ = mat;
            }
            pub fn mulByMat3x2(a: *Mat3x3, b: *Mat3x2) Mat3x2 {
                _ = b;
                _ = a;
            }
            pub fn mulByMat3x3(a: *Mat3x3, b: *Mat3x3) Mat3x3 {
                _ = b;
                _ = a;
            }
        };
    };
}

test "add_vec: unit" {}

test "sub_vec: unit" {}
test "mul_vec: unit" {}

test "mul_mat: unit" {}
test "det_mat: unit" {}
test "inv_mat: unit" {}
test "transpose: unit" {}
