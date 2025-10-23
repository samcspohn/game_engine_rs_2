
mat4 translate(mat4 m, vec3 translation) {
    mat4 t = {
            {
                1,
                0,
                0,
                translation.x
            },
            {
                0,
                1,
                0,
                translation.y
            },
            {
                0,
                0,
                1,
                translation.z
            },
            {
                0,
                0,
                0,
                1
            }
        };
    return m * transpose(t);
}
mat4 translate(vec3 translation) {
    mat4 t = {
            {
                1,
                0,
                0,
                translation.x
            },
            {
                0,
                1,
                0,
                translation.y
            },
            {
                0,
                0,
                1,
                translation.z
            },
            {
                0,
                0,
                0,
                1
            }
        };
    return transpose(t);
}
mat4 scale(mat4 m, vec3 scale) {
    mat4 s = {
            {
                scale.x,
                0,
                0,
                0
            },
            {
                0,
                scale.y,
                0,
                0
            },
            {
                0,
                0,
                scale.z,
                0
            },
            {
                0,
                0,
                0,
                1
            }
        };
    return m * transpose(s);
}
mat4 scale(vec3 scale) {
    mat4 s = {
            {
                scale.x,
                0,
                0,
                0
            },
            {
                0,
                scale.y,
                0,
                0
            },
            {
                0,
                0,
                scale.z,
                0
            },
            {
                0,
                0,
                0,
                1
            }
        };
    return transpose(s);
}

mat4 rotate(mat4 m, vec4 q) {
    mat4 r1 = {
            {
                q.w,
                q.z,
                -q.y,
                q.x
            },
            {
                -q.z,
                q.w,
                q.x,
                q.y
            },
            {
                q.y,
                -q.x,
                q.w,
                q.z
            },
            {
                -q.x,
                -q.y,
                -q.z,
                q.w
            }
        };
    mat4 r2 = {
            {
                q.w,
                q.z,
                -q.y,
                -q.x
            },
            {
                -q.z,
                q.w,
                q.x,
                -q.y
            },
            {
                q.y,
                -q.x,
                q.w,
                -q.z
            },
            {
                q.x,
                q.y,
                q.z,
                q.w
            }
        };
    mat4 r = r1 * r2;
    r[0][3] = r[1][3] = r[2][3] = r[3][0] = r[3][1] = r[3][2] = 0;
    r[3][3] = 1;
    return m * r;
}
mat4 rotate(vec4 q) {
    // mat4 r1 = {{q.w,q.z,-q.y,q.x},{-q.z,q.w,q.x,q.y},{q.y,-q.x,q.w,q.z},{-q.x,-q.y,-q.z,q.w}};
    // mat4 r2 = {{q.w,q.z,-q.y,-q.x},{-q.z,q.w,q.x,-q.y},{q.y,-q.x,q.w,-q.z},{q.x,q.y,q.z,q.w}};
    // mat4 r = r1 * r2;
    // r[0][3] = r[1][3] = r[2][3] = r[3][0] = r[3][1] = r[3][2] = 0;
    // r[3][3] = 1;
    // return r;
    float x = q.x;
    float y = q.y;
    float z = q.z;
    float s = q.w;
    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;

    mat4 r = {
            {
                1.f - 2.f * y2 - 2.f * z2,
                2.f * x * y - 2.f * s * z,
                2.f * x * z + 2.f * s * y,
                0.f
            },
            {
                2.f * x * y + 2.f * s * z,
                1.f - 2.f * x2 - 2.f * z2,
                2.f * y * z - 2.f * s * x,
                0.f
            },
            {
                2.f * x * z - 2.f * s * y,
                2.f * y * z + 2.f * s * x,
                1.f - 2.f * x2 - 2.f * y2,
                0.f
            },
            {
                0.f,
                0.f,
                0.f,
                1.f
            }
        };
    return transpose(r);
}
mat3 rotate3(vec4 q) {
    float qx2 = q.x * q.x;
    float qy2 = q.y * q.y;
    float qz2 = q.z * q.z;
    mat3 r = {
            {
                1 - 2 * qy2 - 2 * qz2,
                2 * q.x * q.y - 2 * q.z * q.w,
                2 * q.x * q.z + 2 * q.y * q.w
            },
            {
                2 * q.x * q.y + 2 * q.z * q.w,
                1 - 2 * qx2 - 2 * qz2,
                2 * q.y * q.z - 2 * q.x * q.w
            },
            {
                2 * q.x * q.z - 2 * q.y * q.w,
                2 * q.y * q.z + 2 * q.x * q.w,
                1 - 2 * qx2 - 2 * qy2
            }
        };
    // r = transpose(r);
    return transpose(r);
}
mat4 identity() {
    mat4 i = {
            {
                1,
                0,
                0,
                0
            },
            {
                0,
                1,
                0,
                0
            },
            {
                0,
                0,
                1,
                0
            },
            {
                0,
                0,
                0,
                1
            }
        };
    return i;
}

vec4 angleAxis(float rad, vec3 axis) {
    vec4 q;
    q.x = axis.x * sin(rad / 2);
    q.y = axis.y * sin(rad / 2);
    q.z = axis.z * sin(rad / 2);
    q.w = cos(rad / 2);
    return q;
}

vec4 RotationBetweenVectors(vec3 start, vec3 dest) {
    start = normalize(start);
    dest = normalize(dest);

    float cosTheta = dot(start, dest);
    vec3 rotationAxis;

    if (cosTheta < -1 + 0.001f) {
        // special case when vectors in opposite directions:
        // there is no "ideal" rotation axis
        // So guess one; any will do as long as it's perpendicular to start
        rotationAxis = cross(vec3(0.0f, 0.0f, 1.0f), start);
        if ((rotationAxis.x * rotationAxis.x + rotationAxis.y * rotationAxis.y + rotationAxis.z * rotationAxis.z) <
                0.01) // bad luck, they were parallel, try again!
            rotationAxis = cross(vec3(1.0f, 0.0f, 0.0f), start);

        rotationAxis = normalize(rotationAxis);
        return angleAxis(radians(180.0f), rotationAxis);
    }

    rotationAxis = cross(start, dest);

    float s = sqrt((1 + cosTheta) * 2);
    float invs = 1 / s;

    return vec4(s * 0.5f, rotationAxis.x * invs, rotationAxis.y * invs, rotationAxis.z * invs);
}

vec4 mat3_quat(mat3 m) {
    #define m00 m[0][0]
    #define m01 m[0][1]
    #define m02 m[0][2]
    #define m10 m[1][0]
    #define m11 m[1][1]
    #define m12 m[1][2]
    #define m20 m[2][0]
    #define m21 m[2][1]
    #define m22 m[2][2]

    float tr = m00 + m11 + m22;
    vec4 q;
    if (tr > 0) {
        float S = sqrt(tr + 1.0) * 2; // S=4*qw
        q.w = 0.25 * S;
        q.x = (m21 - m12) / S;
        q.y = (m02 - m20) / S;
        q.z = (m10 - m01) / S;
    } else if (bool(uint(m00 > m11) & uint(m00 > m22))) {
        float S = sqrt(1.0 + m00 - m11 - m22) * 2; // S=4*qx
        q.w = (m21 - m12) / S;
        q.x = 0.25 * S;
        q.y = (m01 + m10) / S;
        q.z = (m02 + m20) / S;
    } else if (m11 > m22) {
        float S = sqrt(1.0 + m11 - m00 - m22) * 2; // S=4*qy
        q.w = (m02 - m20) / S;
        q.x = (m01 + m10) / S;
        q.y = 0.25 * S;
        q.z = (m12 + m21) / S;
    } else {
        float S = sqrt(1.0 + m22 - m00 - m11) * 2; // S=4*qz
        q.w = (m10 - m01) / S;
        q.x = (m02 + m20) / S;
        q.y = (m12 + m21) / S;
        q.z = 0.25 * S;
    }
    return q;
    #undef m00
    #undef m01
    #undef m02
    #undef m10
    #undef m11
    #undef m12
    #undef m20
    #undef m21
    #undef m22
}
vec4 lookAt(in vec3 lookAt, in vec3 up) {

    // #define m00 right.x
    // #define m01 up.x
    // #define m02 forward.x
    // #define m10 right.y
    // #define m11 up.y
    // #define m12 forward.y
    // #define m20 right.z
    // #define m21 up.z
    // #define m22 forward.z

    vec3 forward = lookAt;
    forward = normalize(forward);
    vec3 right = normalize(cross(up, forward));
    up = normalize(cross(forward, right));

    mat3 m;
    m[0] = right;
    m[1] = up;
    m[2] = forward;
    m = transpose(m);
    // mat3 m = {{right.x,up.x,forward.x},{right.y,up.y,forward.y},{right.z,up.z,forward.z}};
    // transpose(m);
    return mat3_quat(m);
}

mat4 rotationMatrix(vec3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    mat4 r = {
            {
                oc * axis.x * axis.x + c,
                oc * axis.x * axis.y - axis.z * s,
                oc * axis.z * axis.x + axis.y * s,
                0.0
            },
            {
                oc * axis.x * axis.y + axis.z * s,
                oc * axis.y * axis.y + c,
                oc * axis.y * axis.z - axis.x * s,
                0.0
            },
            {
                oc * axis.z * axis.x - axis.y * s,
                oc * axis.y * axis.z + axis.x * s,
                oc * axis.z * axis.z + c,
                0.0
            },
            {
                0.0,
                0.0,
                0.0,
                1.0
            },
        };
    return r;
}

vec3 rotate(vec3 axis, float angle, vec3 vec) {
    if (angle == 0) {
        return vec;
    }
    return (rotationMatrix(axis, angle) * vec4(vec, 1)).xyz;
}
