const std = @import("std");
const mat = @import("matrix.zig");

/// This is a simple Kalman filter
/// n is the number of state parameters to track
/// m is the dimension of the measurements to process, does not need to be the same as the states
pub fn KalmanFilterType(comptime Type: type, n: comptime_int, m: comptime_int) type {
    const nx1 = mat.MatrixType(Type, n, 1);
    const nxn = mat.MatrixType(Type, n, n);
    const mx1 = mat.MatrixType(Type, m, 1);
    const mxn = mat.MatrixType(Type, m, n);
    const mxm = mat.MatrixType(Type, m, m);
    const nxm = mat.MatrixType(Type, n, m);

    const KalmanFilterParams = struct {
        A: ?[n][n]Type, // State transition matrix
        B: ?[n][n]Type, // Control matrix
        H: ?[m][n]Type, // State-to-measurement matrix
        K: ?[n][m]Type, // Kalman Gain
        P: ?[n][n]Type, // State covariance matrix
        Q: ?[n][n]Type, // Process noise covariance matrix
        R: ?[m][m]Type, // Measurement covariance matrix
        u: ?[n][1]Type, // Control vector
        x: ?[n][1]Type, // State variable
    };

    return struct {
        const Filter = @This();

        A: nxn, // State transition matrix
        B: nxn, // Control matrix
        H: mxn, // State-to-measurement matrix
        K: nxm, // Kalman Gain
        P: nxn, // State covariance matrix
        Q: nxn, // Process noise covariance matrix
        R: mxm, // Measurement covariance matrix
        u: nx1, // Control vector
        x: nx1, // State variable

        pub fn init(params: KalmanFilterParams, measurements: [m][]Type) Filter {
            _ = measurements;
            // Initialize the internal state based on the measurements
            var f = Filter{
                .A = params.A orelse nxn.identity(),
                .B = params.B orelse nxn.identity(),
                .H = params.H orelse mxn.identity(),
                .K = params.K orelse nxm.identity(),
                .P = params.P orelse nxn.identity(),
                .Q = params.Q orelse nxn.identity(),
                .R = params.R orelse mxm.identity(),
                .u = params.u orelse nx1.zeros(),
                .x = params.x orelse nx1.zeros(),
            };
            return f;
        }

        // Reinitialize select filter parameters, useful for when knowledge about the system
        // changes over time (external noise, control vector)
        pub fn reinit(
            f: *Filter,
            params: KalmanFilterParams,
        ) Filter {
            var new_f = Filter{
                .A = params.A orelse f.A,
                .B = params.B orelse f.B,
                .H = params.H orelse f.H,
                .K = params.K orelse f.K,
                .P = params.P orelse f.P,
                .Q = params.Q orelse f.Q,
                .R = params.R orelse f.R,
                .u = params.u orelse f.u,
                .x = params.x orelse f.x,
            };
            return new_f;
        }

        pub fn deinit() void {}

        pub fn getState(f: *Filter) struct { nx1, nxn } {
            return .{
                f.x,
                f.P,
            };
        }

        // Predict the next state based on the internal model and covariance matrix
        pub fn predict(f: *Filter) struct { nx1, nxn } {
            var x_p = f.A.mul(&f.x).add(&f.B.mul(&f.u));
            var P_p = f.A.mul(&f.P.mul(&f.A.transpose())).add(&f.Q);

            return .{ x_p, P_p };
        }

        // Update the filter's state with another measurement
        pub fn update(f: *Filter, measurement: [3][1]Type) void {
            var z_k = mx1.init(measurement);
            var H_t = f.H.transpose();
            var stmt = f.H.mul(&f.P.mul(&H_t)).add(f.R).inv();
            var K_p = f.P.mul(&H_t.mul(&stmt));
            var x_p = f.x.add(K_p.mul(&z_k.sub(&f.H.mul(&f.x))));
            var P_p = f.P.sub(&K_p.mul(f.H.mul(&f.P)));

            // Update
            f.x = x_p;
            f.P = P_p;
            f.K = K_p;
        }
    };
}

// Test a filter with position and velocity in 2D. This assumes that measurements are only for the
// position. The prediction matrix is derived from kinematics and assumes the object moves at a 30
// degree angle. The state covariance matrix is the identity since we are using kinematics.
// Although it is possible to derive the initial state solely based on measurements, in this example
// we will set the heading and position
test "kalman" {
    var measurements = [2][]f32{
        [_]f32{ 1, 2, 5, 8 }, // x pos
        [_]f32{ 3, 4, 9, 10 }, // y pos
    };

    var params = .{
        .A = [4][4]f32{
            [4]f32{ 1, 0, @sqrt(3.0) / 2.0, 0 },
            [4]f32{ 0, 1, 0, 0.5 },
            [4]f32{ 0, 0, 1, 0 },
            [4]f32{ 0, 0, 0, 1 },
        },
        .H = [2][4]f32{
            [4]f32{},
            [4]f32{},
        },
        .P = [4][4]f32{
            [4]f32{ 1, 0, 0, 0 },
            [4]f32{ 0, 1, 0, 0 },
            [4]f32{ 0, 0, 2, 0 },
            [4]f32{ 0, 0, 0, 2 },
        },
        .Q = [4][4]f32{
            [4]f32{ 1.5, 0, 0, 0 },
            [4]f32{ 0, 1.5, 0, 0 },
            [4]f32{ 0, 0, 2.5, 0 },
            [4]f32{ 0, 0, 0, 2.5 },
        },
        .R = [2][2]f32{
            [2]f32{ 1.5, 0 },
            [2]f32{ 0, 1.5 },
        },
        .x = [4][1]f32{
            [1]f32{0},
            [1]f32{0},
            [1]f32{1},
            [1]f32{3},
        },
    };

    const kf4x2 = KalmanFilterType(f32, 4, 2);
    var kf = kf4x2.init(
        params,
        measurements,
    );
    _ = kf;
}
