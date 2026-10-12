PLANET_STRIDE = 20

EVENT_CAPACITY = 4096       #max events (merges) between two reads of the cpu

#workgroup size of the force shaders (128 was faster than 256 for most body counts)
NBODY_LOCAL_SIZE = 128

LABEL_CHARS = 12
#characters of the label texture (the digits have to come first)
GLYPHS = "0123456789e-N L,S"

#how many of the strongest attractors of a body get ranked (= max force arrows)
#and the size of a probe record in ints
TOP_ATTRACTORS = 10
PROBE_INTS = 20


VERSION = """#version 430 core
"""

BODY_STRUCT = """
struct Body {
    vec4 pmr;    //position.xy, mass, radius
    vec4 va;     //velocity.xy, acceleration.xy
    vec4 cd;     //color.rgb, density
    vec4 meta;   //type, flags, trail slot, softening
    vec4 lo;     //rest of the position.xy (exact position = pmr.xy + lo.xy)
};

//returns true for background bodies (flag bit 1)
//no force arrows point to them, nothing orbits them and they are drawn smaller
bool is_background(float flags) {
    return (int(flags + 0.5) & 2) != 0;
}

//length of a vector without squaring it directly (the square could overflow a float)
float safe_length(vec2 d) {
    float s = max(abs(d.x), abs(d.y));
    return s > 0.0 ? s * length(d / s) : 0.0;
}
"""

BODIES_READ_WRITE = BODY_STRUCT + """
layout(std430, binding = 0) buffer BodyBuffer { Body bodies[]; };
"""

BODIES_READ_ONLY = BODY_STRUCT + """
layout(std430, binding = 0) readonly buffer BodyBuffer { Body bodies[]; };
"""

#camera data, uploaded once per frame by SimEngine.UploadCamera
#u_anchor = body the camera follows, the center is saved as two floats like the body positions
CAMERA_BLOCK = """
layout(std140, binding = 0) uniform Camera {
    vec2  u_resolution;
    vec2  u_camera_center;
    float u_zoom;
    float u_min_pixel_radius;
    int   u_anchor;
    int   u_body_count;
    vec2  u_view_offset;   //pixels the view center is moved away from the window center
    vec2  u_camera_lo;     //rest of the camera center
};
"""

CAMERA_FUNCTIONS = """
bool valid_body(int i) { return i >= 0 && i < u_body_count; }

vec2 camera_center() {
    return valid_body(u_anchor) ? bodies[u_anchor].pmr.xy : u_camera_center;
}

vec2 camera_lo() {
    return valid_body(u_anchor) ? bodies[u_anchor].lo.xy : u_camera_lo;
}

//converts a world position into screen pixels
vec2 world_to_screen(vec2 world, vec2 cam) {
    return ((world - cam) - camera_lo()) * u_zoom + u_resolution * 0.5 + u_view_offset;
}

//screen position of the center of body i
vec2 body_to_screen(int i, vec2 cam) {
    return world_to_screen(bodies[i].pmr.xy, cam) + bodies[i].lo.xy * u_zoom;
}

vec4 screen_to_clip(vec2 screen) {
    vec2 ndc = screen / u_resolution * 2.0 - 1.0;
    return vec4(ndc.x, -ndc.y, 0.0, 1.0);
}

//drawn radius of a body in pixels (with corona, accretion disc or selection ring)
float visual_radius_px(Body b) {
    int type = int(b.meta.x + 0.5);
    float visual_scale = type == 2 ? 3.3 : (type == 1 ? 1.8 : 1.25);
    return max(u_min_pixel_radius, b.pmr.w * u_zoom) * visual_scale;
}

//position outside the screen, so nothing gets drawn
const vec4 CULLED = vec4(2.0, 2.0, 2.0, 1.0);
"""

QUAD_CORNER = """
//corners of a quad from gl_VertexID (no vertex buffer needed): (-1,-1), (1,-1), (-1,1), (1,1)
vec2 quad_corner() {
    return vec2((gl_VertexID & 1) != 0 ? 1.0 : -1.0, (gl_VertexID & 2) != 0 ? 1.0 : -1.0);
}
"""


#every vertex shader starts with the body buffer and the camera, every fragment shader with the camera
VERTEX_START = VERSION + BODIES_READ_ONLY + CAMERA_BLOCK + CAMERA_FUNCTIONS + QUAD_CORNER
FRAGMENT_START = VERSION + CAMERA_BLOCK


### BODIES ###

VERTEX_SHADER = VERTEX_START + """
out vec3 v_color;
out vec2 v_local_pos;
flat out float v_scale;
flat out int v_type;
flat out int v_locked;

void main() {
    Body body = bodies[gl_InstanceID];

    int type = int(body.meta.x + 0.5);
    int locked = int(body.meta.y + 0.5) & 1;

    //background bodies are drawn smaller when they are tiny on screen
    float min_radius = is_background(body.meta.y) && locked == 0 ? 0.5 * u_min_pixel_radius : u_min_pixel_radius;

    v_color = vec3(0.0);
    v_local_pos = vec2(0.0);
    v_scale = 1.0;
    v_type = type;
    v_locked = 0;

    //absorbed body that waits to be removed
    if (body.pmr.z <= 0.0) {
        gl_Position = CULLED;
        return;
    }

    //stars have a corona and black holes an accretion disc, so their quads have to be bigger
    float scale = 1.0;
    if (type == 2) scale = 3.3;
    else if (type == 1) scale = 1.8;
    if (locked != 0) scale = max(scale, 1.25);

    vec2 corner = quad_corner();
    float display_radius = max(min_radius, body.pmr.w * u_zoom);
    vec2 center = body_to_screen(gl_InstanceID, camera_center());

    gl_Position = screen_to_clip(center + corner * display_radius * scale);

    v_color = body.cd.rgb;
    v_local_pos = corner;
    v_scale = scale;
    v_type = type;
    v_locked = locked;
}
"""


FRAGMENT_SHADER = FRAGMENT_START + """
in vec3 v_color;
in vec2 v_local_pos;
flat in float v_scale;
flat in int v_type;
flat in int v_locked;

out vec4 fragColor;

void main() {
    //distance from the center of the body in body radii
    float dist = length(v_local_pos * v_scale);
    float delta = max(fwidth(dist), 0.0001);

    vec3 rgb;
    float alpha;

    if (v_type == 2) {
        float core = 1.0 - smoothstep(1.0 - delta, 1.0 + delta, dist);
        float ring = exp(-pow((dist - 1.05) / 0.10, 2.0));

        float t = clamp((dist - 1.15) / max(v_scale - 1.15, 0.001), 0.0, 1.0);
        vec3 disc = mix(vec3(1.0, 0.55, 0.0), vec3(0.31, 0.0, 0.02), t);
        float disc_alpha = (1.0 - t) * (1.0 - core) * 0.85;

        rgb = mix(disc, vec3(0.94, 0.96, 1.0), ring);
        rgb = mix(rgb, vec3(0.0), core);
        alpha = max(core, max(disc_alpha, ring));
    } else if (v_type == 1) {
        float core = 1.0 - smoothstep(1.0 - delta, 1.0 + delta, dist);
        float glow = 1.0 - smoothstep(1.0, v_scale, dist);

        rgb = v_color;
        alpha = max(core, glow * glow * 0.55);
    } else {
        rgb = v_color;
        alpha = 1.0 - smoothstep(1.0 - delta, 1.0, dist);
    }

    if (v_locked != 0) {
        //selection ring (a thin line for bodies that are big on screen)
        float select_ring = exp(-pow((dist - 1.1) / min(0.06, 2.0 * delta), 2.0));
        rgb = mix(rgb, vec3(1.0), select_ring);
        alpha = max(alpha, select_ring);
    }

    if (alpha <= 0.002) discard;
    fragColor = vec4(rgb, alpha);
}
"""


### PHYSICS ###

FORCES_SOURCE = BODIES_READ_WRITE + """
layout(local_size_x = __TILE__) in;

uniform int   u_count;
uniform float u_G;
uniform float u_softening;
uniform float u_length_scale;   //see top of this file

shared vec4 tile[__TILE__];   //xy = position, z = G * mass, w = softening squared (scaled)

void main()
{
    uint gid = gl_GlobalInvocationID.x;
    uint lid = gl_LocalInvocationID.x;
    uint count = uint(u_count);
    float scale = u_length_scale;

    vec2 pos = (gid < count) ? bodies[gid].pmr.xy * scale : vec2(0.0);
    vec2 acc = vec2(0.0);

    //the softening gets added to r^2 (kept above zero so a body paired with itself gives 0 and not NaN)
    float soft2 = max(u_softening * scale * scale, 1e-30);
#if SOFT_BODIES
    float own_soft = (gid < count) ? bodies[gid].meta.w * scale : 0.0;
    soft2 += own_soft * own_soft;
#endif

    for (uint base = 0u; base < count; base += __TILE__u)
    {
        uint j = base + lid;
        vec4 other = (j < count) ? bodies[j].pmr : vec4(0.0);
        float soft = 0.0;
#if SOFT_BODIES
        if (j < count) soft = bodies[j].meta.w * scale;
#endif

        //the last tile gets filled up with zero mass
        tile[lid] = vec4(other.xy * scale, u_G * other.z * (scale * scale), soft * soft);
        barrier();

        for (uint k = 0u; k < __TILE__u; ++k)
        {
#if SOFT_BODIES
            vec4 t = tile[k];
            vec2 d = t.xy - pos;
            float inv_r = inversesqrt(dot(d, d) + soft2 + t.w);
#else
            vec3 t = tile[k].xyz;
            vec2 d = t.xy - pos;
            float inv_r = inversesqrt(dot(d, d) + soft2);
#endif

            //(Gm / r) / r / r instead of Gm / r^3 because 1 / r^3 gets too small for a float
            float s = t.z * inv_r;
            acc += d * (s * inv_r * inv_r);
        }

        barrier();
    }

    if (gid < count) bodies[gid].va.zw = acc;
}
""".replace("__TILE__", str(NBODY_LOCAL_SIZE))

NBODY_FORCES = VERSION + "#define SOFT_BODIES 0\n" + FORCES_SOURCE
NBODY_FORCES_SOFT = VERSION + "#define SOFT_BODIES 1\n" + FORCES_SOURCE

STEP_LIMIT = VERSION + BODIES_READ_ONLY + """
layout(local_size_x = __TILE__) in;

layout(std430, binding = 22) buffer StepLimitBuffer { uint fastest; };   //in s^-2, saved as float bits

uniform int   u_count;
uniform float u_G;
uniform float u_softening;
uniform float u_length_scale;   //see top of this file

shared vec4 tile[__TILE__];   //xy = position, z = G * mass, w = softening squared (scaled)

void main()
{
    uint gid = gl_GlobalInvocationID.x;
    uint lid = gl_LocalInvocationID.x;
    uint count = uint(u_count);
    float scale = u_length_scale;

    vec4 self_pmr = (gid < count) ? bodies[gid].pmr : vec4(0.0);
    vec2 pos = self_pmr.xy * scale;
    float own = u_G * self_pmr.z * (scale * scale);
    float own_soft = (gid < count) ? bodies[gid].meta.w * scale : 0.0;
    float soft2 = max(u_softening * scale * scale, 1e-30) + own_soft * own_soft;
    uint own_tile = gl_WorkGroupID.x * __TILE__u;
    float quickest = 0.0;

    for (uint base = 0u; base < count; base += __TILE__u)
    {
        uint j = base + lid;
        vec4 other = (j < count) ? bodies[j].pmr : vec4(0.0);
        float soft = (j < count) ? bodies[j].meta.w * scale : 0.0;
        tile[lid] = vec4(other.xy * scale, u_G * other.z * (scale * scale), soft * soft);
        barrier();

        for (uint k = 0u; k < __TILE__u; ++k)
        {
            vec4 t = tile[k];
            vec2 d = t.xy - pos;
            float inv_r = inversesqrt(dot(d, d) + soft2 + t.w);

            //no pair with itself, with absorbed bodies or with the padding
            bool pair = t.z > 0.0 && !(base == own_tile && k == lid);
            quickest = max(quickest, pair ? (own + t.z) * inv_r * inv_r * inv_r : 0.0);
        }

        barrier();
    }

    //corrects the scaling and saves the biggest value (positive floats sort like their bits)
    if (gid < count && own > 0.0) atomicMax(fastest, floatBitsToUint(quickest * scale));
}
""".replace("__TILE__", str(NBODY_LOCAL_SIZE))


### COLLISIONS AND MERGING ###

#constants and star formulas the gpu needs (same as in gpu_cso.py)
STAR_FORMULAS = """
const float SUN_MASS = 1.989e30;                //kg
const float SUN_RADIUS = 6.957e8;               //m
const float AU_IN_METERS = 149597870700.0;

//part of the mass that gets radiated away when matter falls into a black hole: 1 - sqrt(8/9)
const float ISCO_EFFICIENCY = 0.0571909584;

//radius of a main sequence star from its mass (same as calculate_star_radius in gpu_cso.py)
float stellar_radius(float mass) {
    float m = mass / SUN_MASS;
    float r = m < 0.43 ? pow(m, 0.95) : (m < 1.0 ? pow(m, 0.8) : (m < 5.0 ? pow(m, 0.57) : pow(m, 0.72)));
    return r * SUN_RADIUS;
}

//luminosity in sun luminosities (same as calculate_luminosity in gpu_cso.py)
float luminosity(float mass) {
    float m = mass / SUN_MASS;
    if (m < 0.43) return 0.19 * pow(m, 2.3);
    if (m < 2.0) return pow(m, 4.0);
    if (m < 20.0) return 1.4 * pow(m, 3.5);
    return 14500.0 * m;
}
"""

AUX_BUFFER = """
struct Aux {
    int   partner;       //body this one touches and gets absorbed by (-1 = none)
    int   claim;
    int   tide_source;   //body with the strongest tides on this one (-1 = none)
    int   pad0;
    float tidal;         //G m / r^3 of that body (s^-2) ...
    float tidal_r2;      //... and its distance squared (m^2)
    float pad1;
    float pad2;
};
layout(std430, binding = 18) buffer AuxBuffer { Aux aux[]; };

const int NO_CLAIM = 2147483647;
"""

#roche distance of a body to the body causing its tides: d (2 T R^3 / (G m))^(1/3) with T = G M / d^3
ROCHE_FUNCTION = """
float roche_distance(float tidal, float d, float radius, float gm) {
    return d * pow(2.0 * tidal * radius * radius * radius / gm, 1.0 / 3.0);
}
"""

#merge events for the cpu
EVENT_BUFFER = """
struct Event {
    ivec4 info;    //x = kind (1 = merge), y = uid of the survivor, z = uid of the absorbed body
    vec4  data;    //new mass, radius, density, radiated mass
};
layout(std430, binding = 17) buffer EventBuffer {
    uint  event_count;
    uint  event_pad0, event_pad1, event_pad2;
    Event events[];
};

const int EVENT_MERGE = 1;
const uint EVENT_CAPACITY = __EVENT_CAPACITY__u;

void emit_event(ivec4 info, vec4 data) {
    uint slot = atomicAdd(event_count, 1u);
    if (slot < EVENT_CAPACITY) events[slot] = Event(info, data);
}
""".replace("__EVENT_CAPACITY__", str(EVENT_CAPACITY))

CONTACT_SOURCE = BODIES_READ_WRITE + AUX_BUFFER + """
layout(local_size_x = __TILE__) in;

layout(std430, binding = 20) readonly buffer BlackHoleBuffer { int black_holes[]; };

uniform int   u_count;
uniform float u_G;
uniform float u_softening;
uniform float u_length_scale;   //see top of this file
uniform float u_dt;
uniform int   u_black_hole_count;
uniform int   u_merge;          //0 while paused (bodies only merge while the sim runs)
uniform float u_c;              //speed of light (m/s)

shared vec4  tile[__TILE__];        //xy = position, z = G * mass, w = contact radius / 2 (scaled)
#if SOFT_BODIES
shared float tile_soft[__TILE__];   //softening squared (scaled)
#endif

const float BLACK_HOLE_RANK = 1024.0;   //ranks from here on are black holes

//decides who absorbs whom (black holes before stars before planets, then the heavier one)
float merge_rank(float mass, float type) {
    return mass > 0.0 ? floor(type + 0.5) * 512.0 + log2(mass) : -1e30;
}

//a black hole absorbs everything within 3 times its horizon, all other bodies touch at their surface
float contact_radius(float radius, float type) {
    return type > 1.5 ? 3.0 * radius : radius;
}

//distance below which two passing black holes lose enough energy by gravitational waves to merge
//(Quinlan & Shapiro 1989), calculated with logarithms because the values overflow a float
float capture_radius(float m1, float m2, float v2) {
    float l = log2(85.0 * 3.14159265 * 1.41421356 / 12.0) + 3.5 * log2(u_G)
            + log2(m1) + log2(m2) + 1.5 * log2(m1 + m2)
            - 5.0 * log2(u_c) - log2(max(v2, 1.0));
    return exp2(l * (2.0 / 7.0));
}

//adds the pull of one tile entry and remembers the strongest tide
//returns a value below 0 when the two bodies touch
float interact(vec4 t, vec2 pos, float half_radius, float soft2, inout vec2 acc, inout float tidal) {
    vec2 d = t.xy - pos;
    float r2 = dot(d, d);
    float mean_radius = half_radius + t.w;
    float mean2 = mean_radius * mean_radius;

    //overlapping bodies: the pull stops growing at their mean radius
    float inv_r = inversesqrt(max(r2 + soft2, mean2));
    float s = t.z * inv_r;
    float s3 = s * inv_r * inv_r;
    acc += d * s3;

    //tides only count for bodies that dont touch
    float gap = 0.25 * r2 - mean2;
    tidal = max(tidal, gap < 0.0 ? 0.0 : s3);
    return gap;
}

//returns the body that absorbs body i (or -1): of the bodies it touches the one with the highest rank
//black holes are checked along the whole path of the sub step because they are small and fast
//two black holes already merge when they pass within their capture distance
int absorber(uint i, vec4 self_pmr, float radius, float rank, float contact, uint contact_base, float soft2)
{
    uint count = uint(u_count);
    vec2 pos = self_pmr.xy;
    int partner = -1;
    float partner_rank = -1e30;

    if (contact < 0.0) {
        for (uint j = contact_base; j < min(contact_base + __TILE__u, count); ++j) {
            if (j == i) continue;
            vec4 other = bodies[j].pmr;
            float other_type = bodies[j].meta.x;
            vec2 d = other.xy - pos;
            float reach = radius + contact_radius(other.w, other_type);
            if (dot(d, d) >= reach * reach) continue;

            float other_rank = merge_rank(other.z, other_type);
            bool beats_self = other_rank > rank || (other_rank == rank && int(j) > int(i));
            bool beats_best = partner < 0 || other_rank > partner_rank
                           || (other_rank == partner_rank && int(j) > partner);
            if (beats_self && beats_best) {
                partner = int(j);
                partner_rank = other_rank;
            }
        }
    }

    vec2 velocity = bodies[i].va.xy;
    float self_gm = u_G * self_pmr.z;
    bool black_hole = rank >= BLACK_HOLE_RANK;
    for (int n = 0; n < u_black_hole_count; ++n) {
        int j = black_holes[n];
        if (j < 0 || j >= int(count) || j == int(i)) continue;
        vec4 b = bodies[j].pmr;
        float other_rank = merge_rank(b.z, 2.0);
        if (!(other_rank > rank || (other_rank == rank && j > int(i)))) continue;

        vec2 d = b.xy - pos;
        float reach = radius + 3.0 * b.w;
        float inv_r = inversesqrt(max(dot(d, d) + soft2, 0.25 * reach * reach));
        float pull = (u_G * b.z + self_gm) * inv_r * inv_r * inv_r;
        vec2 dv = bodies[j].va.xy - velocity - d * (pull * u_dt);
        float dd = dot(dv, dv);
        float when = dd > 0.0 ? clamp(-dot(d, dv) / dd, 0.0, u_dt) : 0.0;
        vec2 closest = d + dv * when;
        float capture = black_hole ? max(reach, capture_radius(self_pmr.z, b.z, dd)) : reach;

        if (dot(closest, closest) < capture * capture
                && (partner < 0 || other_rank > partner_rank || (other_rank == partner_rank && j > partner))) {
            partner = j;
            partner_rank = other_rank;
        }
    }
    return partner;
}

//returns the body with the strongest tides on body i and its squared distance
int strongest_tide(uint i, vec2 pos, float half_radius, float soft2, uint tidal_base, out float distance2)
{
    uint count = uint(u_count);
    int source = -1;
    float strongest = 0.0;
    distance2 = 0.0;
    for (uint j = tidal_base; j < min(tidal_base + __TILE__u, count); ++j) {
        if (j == i) continue;
        vec4 other = bodies[j].pmr;
        vec2 d = other.xy - pos;
        float r2 = dot(d, d);
        float mean_radius = half_radius + 0.5 * contact_radius(other.w, bodies[j].meta.x);
        float mean2 = mean_radius * mean_radius;
        if (0.25 * r2 < mean2) continue;
        float inv_r = inversesqrt(max(r2 + soft2, mean2));
        float s3 = u_G * other.z * inv_r * inv_r * inv_r;
        if (s3 > strongest) {
            strongest = s3;
            source = int(j);
            distance2 = r2;
        }
    }
    return source;
}

void main()
{
    uint gid = gl_GlobalInvocationID.x;
    uint lid = gl_LocalInvocationID.x;
    uint count = uint(u_count);
    bool valid = gid < count;
    float scale = u_length_scale;

    vec4 self_pmr = valid ? bodies[gid].pmr : vec4(0.0);
    vec4 self_meta = valid ? bodies[gid].meta : vec4(0.0);
    vec2 pos = self_pmr.xy * scale;
    float radius = contact_radius(self_pmr.w, self_meta.x);
    float rank = valid ? merge_rank(self_pmr.z, self_meta.x) : -1e30;

    vec2 acc = vec2(0.0);
    float tidal = 0.0;          //strongest tide (scaled) ...
    uint tidal_base = 0u;       //... and the tile it came from
    float contact = 0.0;        //deepest overlap (below 0 = touches something) ...
    uint contact_base = 0u;     //... and the tile it was found in

    float soft2 = max(u_softening * scale * scale, 1e-30);
#if SOFT_BODIES
    float own_soft = self_meta.w * scale;
    soft2 += own_soft * own_soft;
    #define PAIR_SOFT2(k) (soft2 + tile_soft[k])
#else
    #define PAIR_SOFT2(k) soft2
#endif
    float half_radius = 0.5 * radius * scale;
    uint own_tile = gl_WorkGroupID.x * __TILE__u;

    for (uint base = 0u; base < count; base += __TILE__u)
    {
        uint j = base + lid;
        bool real = j < count;
        vec4 other = real ? bodies[j].pmr : vec4(0.0);
        vec4 other_meta = real ? bodies[j].meta : vec4(0.0);

        //padding is placed far away so nothing touches it
        tile[lid] = real ? vec4(other.xy * scale, u_G * other.z * (scale * scale), 0.5 * contact_radius(other.w, other_meta.x) * scale)
                         : vec4(3e38, 3e38, 0.0, 0.0);
#if SOFT_BODIES
        tile_soft[lid] = other_meta.w * scale * other_meta.w * scale;
#endif
        barrier();

        float before = contact;
        float tidal_before = tidal;

        //only in the tile of the body itself the body has to be left out
        if (base == own_tile) {
            for (uint k = 0u; k < __TILE__u; ++k) {
                float gap = interact(tile[k], pos, half_radius, PAIR_SOFT2(k), acc, tidal);
                contact = min(contact, k == lid ? 0.0 : gap);
            }
        } else {
            for (uint k = 0u; k < __TILE__u; ++k)
                contact = min(contact, interact(tile[k], pos, half_radius, PAIR_SOFT2(k), acc, tidal));
        }

        contact_base = contact < before ? base : contact_base;
        tidal_base = tidal > tidal_before ? base : tidal_base;
        barrier();
    }

    if (!valid) return;

    bool alive = self_pmr.z > 0.0;
    float metres_soft2 = max(u_softening, 1e-6);
    int partner = alive && u_merge != 0 ? absorber(gid, self_pmr, radius, rank, contact, contact_base, metres_soft2)
                                        : -1;

    //the body with the strongest tides is only looked up when the body could be near a roche limit
    tidal *= scale;
    int tide_source = -1;
    float tidal_r2 = 0.0;
    float R = self_pmr.w;
    if (alive && 2.0 * tidal * R * R * R / (u_G * self_pmr.z) > 0.04)
        tide_source = strongest_tide(gid, self_pmr.xy, 0.5 * radius, metres_soft2, tidal_base, tidal_r2);

    bodies[gid].va.zw = acc;
    aux[gid].partner = partner;
    aux[gid].tide_source = tide_source;
    aux[gid].tidal = tidal;
    aux[gid].tidal_r2 = tidal_r2;
    if (partner >= 0) atomicMin(aux[partner].claim, int(gid));
}
""".replace("__TILE__", str(NBODY_LOCAL_SIZE))

NBODY_FORCES_CONTACT = VERSION + "#define SOFT_BODIES 0\n" + CONTACT_SOURCE
NBODY_FORCES_CONTACT_SOFT = VERSION + "#define SOFT_BODIES 1\n" + CONTACT_SOURCE

NBODY_MERGE = VERSION + BODIES_READ_WRITE + AUX_BUFFER + EVENT_BUFFER + STAR_FORMULAS + """
layout(local_size_x = 256) in;

uniform int   u_count;
uniform float u_G;
uniform float u_c;

const float MASS_LIMIT = 3.0e38;    //kg (the biggest float is 3.4e38)

void main()
{
    int p = int(gl_GlobalInvocationID.x);
    if (p >= u_count) return;

    int c = aux[p].claim;
    if (c == NO_CLAIM) return;
    aux[p].claim = NO_CLAIM;
    if (aux[p].partner >= 0 || c < 0 || c >= u_count) return;

    Body a = bodies[p];
    Body b = bodies[c];
    float ma = a.pmr.z;
    float mb = b.pmr.z;
    if (ma <= 0.0 || mb <= 0.0) return;

    float total = min(ma + mb, MASS_LIMIT);
    float f = mb / total;

    //center of mass (with the exact positions)
    vec2 delta = (b.pmr.xy - a.pmr.xy) + (b.lo.xy - a.lo.xy);
    vec2 t = a.lo.xy + delta * f;
    precise vec2 s = a.pmr.xy + t;
    precise vec2 bb = s - a.pmr.xy;
    precise vec2 e = (a.pmr.xy - (s - bb)) + (t - bb);

    int type_a = int(a.meta.x + 0.5);
    int type_b = int(b.meta.x + 0.5);

    float mass = total;
    float radiated = 0.0;
    float radius;
    float density = a.cd.w;

    if (type_a == 2) {
        if (type_b == 2) {
            float nu = (ma / total) * (mb / total);
            radiated = total * (0.0572 * nu + 0.5435 * nu * nu);
        } else {
            radiated = ISCO_EFFICIENCY * mb;
        }
        mass = total - radiated;
        radius = 2.0 * u_G * mass / (u_c * u_c);
    } else if (type_a == 1) {
        radius = stellar_radius(mass);
        density = mass / (4.18879020 * radius * radius * radius);
    } else {
        float volume = ma / max(a.cd.w, 1e-12) + mb / max(b.cd.w, 1e-12);
        density = mass / volume;
        radius = pow(volume / 4.18879020, 1.0 / 3.0);
    }

    bodies[p].pmr = vec4(s, mass, radius);
    bodies[p].lo.xy = e;
    bodies[p].va = a.va + (b.va - a.va) * f;
    bodies[p].cd.w = density;

    bodies[c].pmr = vec4(b.pmr.xy, 0.0, 0.0);
    bodies[c].va = vec4(0.0);

    emit_event(ivec4(EVENT_MERGE, int(a.lo.z), int(b.lo.z), 0), vec4(mass, radius, density, radiated));
}
"""


NBODY_INTEGRATE = VERSION + BODIES_READ_WRITE + """
layout(local_size_x = 256) in;

layout(std430, binding = 1) writeonly buffer TrailBuffer {
    vec2 trail_data[];
};

uniform int   u_count;
uniform float u_dt;
uniform int   u_trail_length;
uniform int   u_trail_head;
uniform int   u_write_trail;

void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_count)) return;

    vec4 va = bodies[i].va;
    vec2 hi = bodies[i].pmr.xy;
    vec2 lo = bodies[i].lo.xy;

    va.xy += va.zw * u_dt;

    //adds the step to the small rest first and then carries it into the rounded position without losing precision
    //(precise stops the compiler from simplifying this away)
    vec2 t = lo + va.xy * u_dt;
    precise vec2 s = hi + t;
    precise vec2 b = s - hi;
    precise vec2 e = (hi - (s - b)) + (t - b);

    bodies[i].pmr.xy = s;
    bodies[i].lo.xy = e;
    bodies[i].va.xy = va.xy;

    if (u_write_trail != 0) {
        int slot = int(bodies[i].meta.z);
        if (slot >= 0) trail_data[slot * u_trail_length + u_trail_head] = s;
    }
}
"""


KEPLER_STEP = """
const float KEPLER_TURN = 0.05;     //part of an orbit (rad) per step from which the exact orbit is used

//moves a particle along its two-body orbit for dt (r and v relative to a body with G M = mu)
//returns false if the orbit is not a closed ellipse (then a normal step is used)
bool kepler_step(inout vec2 r, inout vec2 v, float mu, float dt)
{
    float r0 = length(r);
    float h = r.x * v.y - r.y * v.x;        //angular momentum
    float inv_a = 2.0 / r0 - dot(v, v) / mu;
    if (!(inv_a > 0.0)) return false;
    float a = 1.0 / inv_a;
    float sqrt_mu_a = sqrt(mu) * sqrt(a);
    float ec = 1.0 - r0 * inv_a;            //e cos E0
    float es = dot(r, v) / sqrt_mu_a;       //e sin E0
    float e = length(vec2(ec, es));
    if (e > 0.99) return false;

    float n = sqrt(mu * inv_a) * inv_a;     //mean motion, sqrt(mu / a^3)
    float M = mod(n * dt, 6.2831853);       //full turns change nothing

    //solves Kepler's equation for x: x - ec sin x + es (1 - cos x) = M
    float E0 = atan(es, ec);
    float M1 = E0 - es + M;
    float x = M1 + 0.85 * e * (sin(M1) < 0.0 ? -1.0 : 1.0) - E0;
    for (int k = 0; k < 8; ++k) {
        float s = sin(x), c = cos(x);
        x -= (x - ec * s + es * (1.0 - c) - M) / (1.0 - ec * c + es * s);
    }

    float s = sin(x), c = cos(x);
    float f = 1.0 - a / r0 * (1.0 - c);
    float g = (M - x + s) / n;
    float r1 = a + (r0 - a) * c + dot(r, v) / sqrt(mu) * sqrt(a) * s;
    float f_dot = -sqrt_mu_a * s / (r1 * r0);
    float g_dot = 1.0 - a / r1 * (1.0 - c);

    vec2 r_new = f * r + g * v;
    v = f_dot * r + g_dot * v;
    r = r_new;

    //rounding errors on the gpu would slowly shrink every orbit,
    //so the sideways speed gets corrected to keep the angular momentum exactly
    v += (h - (r.x * v.y - r.y * v.x)) / dot(r, r) * vec2(-r.y, r.x);
    return true;
}

//returns true if a step of dt is too long for an orbit at distance r
bool coarse_step(float mu, float r, float dt) {
    return r > 0.0 && sqrt(mu / r) / r * dt > KEPLER_TURN;
}
"""


#particlesget pulled by the bodies but have no pull themselves
PARTICLE_CALCULATION = VERSION + BODIES_READ_ONLY + KEPLER_STEP + """
layout(local_size_x = 256) in;

layout(std430, binding = 2) buffer ParticleBuffer {
    vec4 particles[];   //xy = position, zw = velocity (relative to the host if attached)
};

layout(std430, binding = 12) readonly buffer ParticleGroupBuffer {
    ivec4 particle_groups[];   //x = first particle, y = count, z = host body
};

uniform int   u_planet_count;
uniform int   u_count;
uniform int   u_group_count;
uniform float u_dt;
uniform float u_G;
uniform float u_softening;
uniform float u_length_scale;   //see top of this file

shared vec4 tile[256];      //xy = position, z = G * mass, w = softening squared (scaled)

//moves the particle along its orbit around the body center (see KEPLER_STEP), the other bodies pull with other_acc
//returns false if a normal step is good enough or the orbit is not closed
bool orbit_step(inout vec4 state, int center, bool attached, vec2 other_acc)
{
    vec4 b = bodies[center].pmr;
    float mu = u_G * b.z;

    //attached particles are already relative to the host, free ones get converted to be relative to the body
    vec4 va = bodies[center].va;
    vec2 frame_v = attached ? vec2(0.0) : va.xy + va.zw * u_dt;
    vec2 r = attached ? state.xy : (state.xy - b.xy) - bodies[center].lo.xy;
    float distance = length(r);
    if (distance * distance < 25.0 * u_softening || !coarse_step(mu, distance, u_dt)) return false;
    vec2 v = state.zw + (other_acc - (attached ? va.zw : vec2(0.0))) * u_dt - frame_v;

    if (!kepler_step(r, v, mu, u_dt)) return false;

    vec2 center_now = attached ? vec2(0.0) : (b.xy + bodies[center].lo.xy) + frame_v * u_dt;
    state = vec4(center_now + r, frame_v + v);
    return true;
}

void main()
{
    uint gid = gl_GlobalInvocationID.x;
    uint lid = gl_LocalInvocationID.x;
    uint planets = uint(u_planet_count);
    bool in_range = gid < uint(u_count);
    float scale = u_length_scale;

    int host = -1;
    for (int g = 0; g < u_group_count; ++g) {
        ivec4 group = particle_groups[g];
        if (int(gid) >= group.x && int(gid) < group.x + group.y) {
            host = group.z;
            break;
        }
    }

    //free particles have the origin 0
    bool attached = host >= 0 && host < int(planets);
    vec2 origin = attached ? bodies[host].pmr.xy : vec2(0.0);
    vec2 frame_acc = attached ? bodies[host].va.zw : vec2(0.0);

    vec4 state = in_range ? particles[gid] : vec4(0.0);
    bool leap = attached && !coarse_step(u_G * bodies[host].pmr.z, length(state.xy), u_dt);
    if (leap) state.xy += state.zw * (0.5 * u_dt);
    vec2 origin_scaled = origin * scale;
    vec2 here_scaled = state.xy * scale;
    float soft2 = max(u_softening * scale * scale, 1e-30);

    //the pull of the orbited body is kept separate from the pull of all others (the orbit step needs both)
    vec2 acc = vec2(0.0);       //all bodies except the host
    vec2 host_acc = vec2(0.0);
    vec2 strongest_acc = vec2(0.0);
    float strongest = 0.0;
    int dominant = -1;

    for (uint base = 0u; base < planets; base += 256u)
    {
        uint j = base + lid;
        vec4 other = (j < planets) ? bodies[j].pmr : vec4(0.0);
        float soft = (j < planets) ? bodies[j].meta.w * scale : 0.0;
        tile[lid] = vec4(other.xy * scale, u_G * other.z * (scale * scale), soft * soft);
        barrier();

        uint n = min(256u, planets - base);
        for (uint k = 0u; k < n; ++k)
        {
            vec4 t = tile[k];
            vec2 d = (t.xy - origin_scaled) - here_scaled;
            float inv_r = inversesqrt(dot(d, d) + soft2 + t.w);
            float s = t.z * inv_r;
            vec2 a = d * (s * inv_r * inv_r);

            bool own = int(base + k) == host;
            acc += own ? vec2(0.0) : a;
            host_acc += own ? a : vec2(0.0);

            float pull = s * inv_r;
            bool stronger = pull > strongest;
            dominant = stronger ? int(base + k) : dominant;
            strongest_acc = stronger ? a : strongest_acc;
            strongest = max(strongest, pull);
        }

        barrier();
    }

    if (!in_range) return;

    if (leap) {
        state.zw += (acc + host_acc - frame_acc) * u_dt;
        state.xy += state.zw * (0.5 * u_dt);
    } else {
        bool along_orbit = attached ? orbit_step(state, host, true, acc)
                         : dominant >= 0 && orbit_step(state, dominant, false, acc - strongest_acc);
        if (!along_orbit) {
            state.zw += (acc + host_acc - frame_acc) * u_dt;
            state.xy += state.zw * u_dt;
        }
    }
    particles[gid] = state;
}
"""


#turns attached particles back into free particles (when their host gets deleted)
PARTICLE_DETACH = VERSION + BODIES_READ_ONLY + """
layout(local_size_x = 256) in;

layout(std430, binding = 2) buffer ParticleBuffer {
    vec4 particles[];
};

uniform int u_first;
uniform int u_count;
uniform int u_host;

void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_count)) return;

    uint p = uint(u_first) + i;
    particles[p] += vec4(bodies[u_host].pmr.xy, bodies[u_host].va.xy);
}
"""


### UPDATE BUFFER ###
#the only way changes from the cpu get into the body buffer

UPDATE_COMMON = """
layout(local_size_x = 64) in;

struct Update {
    ivec4 header;              //x = destination, y = source or -1, z = field mask
    float payload[__STRIDE__];
};

uniform int u_update_count;
"""

UPDATE_GATHER = """
layout(std430, binding = __TARGET__) readonly buffer Target { float target_data[]; };
layout(std430, binding = __UPDATES__) buffer Updates { Update updates[]; };

void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_update_count)) return;

    int source = updates[i].header.y;
    if (source < 0) return;

    int mask = updates[i].header.z;
    uint base = uint(source) * __STRIDE__u;

    //fields that dont get changed come from the source record
    for (int f = 0; f < __STRIDE__; ++f)
        if ((mask & (1 << f)) == 0) updates[i].payload[f] = target_data[base + uint(f)];

    updates[i].header.z = __FULL_MASK__;
}
"""

UPDATE_SCATTER = """
layout(std430, binding = __TARGET__) buffer Target { float target_data[]; };
layout(std430, binding = __UPDATES__) readonly buffer Updates { Update updates[]; };

void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_update_count)) return;

    int mask = updates[i].header.z;
    uint base = uint(updates[i].header.x) * __STRIDE__u;

    for (int f = 0; f < __STRIDE__; ++f)
        if ((mask & (1 << f)) != 0) target_data[base + uint(f)] = updates[i].payload[f];
}
"""


def update_shaders(stride, target_binding, update_binding):
    assert stride % 4 == 0 and stride <= 31

    shaders = []
    for source in (UPDATE_GATHER, UPDATE_SCATTER):
        source = VERSION + UPDATE_COMMON + source
        source = source.replace("__STRIDE__", str(stride))
        source = source.replace("__FULL_MASK__", str((1 << stride) - 1))
        source = source.replace("__TARGET__", str(target_binding))
        source = source.replace("__UPDATES__", str(update_binding))
        shaders.append(source)

    return shaders[0], shaders[1]


#sets all points of a new trail to the current position of its body
TRAIL_RESET = VERSION + BODIES_READ_ONLY + """
layout(local_size_x = 256) in;

layout(std430, binding = 1) writeonly buffer TrailBuffer {
    vec2 trail_data[];
};

layout(std430, binding = 10) readonly buffer TrailResetBuffer {
    ivec2 trail_resets[];   //x = slot, y = body index
};

uniform int u_reset_count;
uniform int u_trail_length;

void main()
{
    uint t = gl_GlobalInvocationID.x;
    uint trail_length = uint(u_trail_length);
    uint k = t / trail_length;
    if (k >= uint(u_reset_count)) return;

    ivec2 reset = trail_resets[k];
    trail_data[uint(reset.x) * trail_length + t % trail_length] = bodies[reset.y].pmr.xy;
}
"""


### PARTICLES AND TRAILS ###

#drawn as points, one vertex per particle
PARTICLE_VERTEX_SHADER = VERTEX_START + """
layout(std430, binding = 2) readonly buffer ParticleBuffer {
    vec4 particles[];
};

uniform int u_first;
uniform int u_host;   //body the particles are relative to (-1 = none)

void main() {
    vec2 pos = particles[u_first + gl_VertexID].xy;
    vec2 cam = camera_center();

    //attached particles: host position minus camera first, so they stay exact
    vec2 screen = valid_body(u_host)
        ? body_to_screen(u_host, cam) + pos * u_zoom
        : world_to_screen(pos, cam);

    gl_Position = screen_to_clip(screen);
}
"""


PARTICLE_FRAGMENT_SHADER = FRAGMENT_START + """
uniform vec4 u_color;

out vec4 fragColor;

void main() {
    fragColor = u_color;
}
"""


TRAIL_VERTEX_SHADER = VERTEX_START + """
layout(std430, binding = 1) readonly buffer TrailBuffer {
    vec2 trail_data[];
};

struct TrailMeta {
    vec4  color;   //rgb = trail color
    ivec4 info;    //x = active, y = recorded points when the trail started
};

layout(std430, binding = 3) readonly buffer TrailMetaBuffer {
    TrailMeta trail_meta[];
};

uniform int u_trail_length;
uniform int u_trail_head;
uniform int u_total_recorded;

out float v_alpha;
out vec3 v_color;

void main() {
    int slot = gl_InstanceID;
    TrailMeta meta = trail_meta[slot];

    v_color = meta.color.rgb;
    v_alpha = 0.0;

    if (meta.info.x == 0) {
        gl_Position = CULLED;
        return;
    }

    //gl_VertexID goes from the oldest point (0) to the newest, the newest one is at u_trail_head
    int age = (u_trail_length - 1) - gl_VertexID;
    int available = clamp(u_total_recorded - meta.info.y + 1, 1, u_trail_length);

    if (age < available) {
        v_alpha = float(u_trail_length - age) / float(u_trail_length);
    } else {
        //not recorded yet: these points are put onto the oldest recorded one
        age = available - 1;
    }

    int ring_index = (u_trail_head - age + u_trail_length) % u_trail_length;
    vec2 pos = trail_data[slot * u_trail_length + ring_index];

    gl_Position = screen_to_clip(world_to_screen(pos, camera_center()));
}
"""


TRAIL_FRAGMENT_SHADER = FRAGMENT_START + """
in float v_alpha;
in vec3 v_color;

out vec4 fragColor;

void main() {
    if (v_alpha <= 0.01) discard;
    fragColor = vec4(v_color, v_alpha);
}
"""


### CLICKING ON BODIES ###

PICK_CALCULATION = VERSION + BODIES_READ_ONLY + CAMERA_BLOCK + CAMERA_FUNCTIONS + """
layout(local_size_x = 256) in;

layout(std430, binding = 4) buffer PickBuffer {
    uint pick_result[];
};

uniform vec2  u_mouse;
uniform float u_pick_radius;

void main()
{
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uint(u_body_count)) return;

    vec4 body = bodies[gid].pmr;
    if (body.z <= 0.0) return;
    vec2 screen_pos = body_to_screen(int(gid), camera_center());

    //tiny bodies still get a big enough click area
    float radius = max(u_pick_radius, max(u_min_pixel_radius, body.w * u_zoom));
    float dist = distance(screen_pos, u_mouse);

    if (!(dist <= radius)) return;

    //background bodies only get picked if no other body is in reach
    uint quantised = uint(clamp(dist, 0.0, 2047.0));
    if (is_background(bodies[gid].meta.y)) quantised += 2048u;
    uint key = (quantised << 20) | (gid & 0xFFFFFu);

    atomicMin(pick_result[0], key);
}
"""


### OVERLAYS ###
#everything below reads its data from the gpu, the cpu only uploads which body has which overlay

PROBE_STRUCT = """
const int TOP_ATTRACTORS = __TOP__;

struct Probe {
    ivec4 info;             //x = target body (written by the cpu)
    ivec4 extra;            //x = body it orbits (see INSPECT_CALCULATION)
    int   top[__SLOTS__];   //strongest attractors, strongest first (-1 = none)
};

//a body that nothing heavier holds counts as orbiting its strongest attractor,
//if that one has at least this share of its mass
const float COMPANION = 0.1;
""".replace("__TOP__", str(TOP_ATTRACTORS)).replace("__SLOTS__", str(PROBE_INTS - 8))

INSPECT_CALCULATION = VERSION + BODIES_READ_ONLY + PROBE_STRUCT + """
layout(local_size_x = 256) in;

layout(std430, binding = 5) buffer ProbeBuffer { Probe probes[]; };

uniform int u_count;

shared float s_value[256];
shared int   s_index[256];
shared int   s_owner[256];

//m / r^2 of other at the position at, without calculating r^2 directly (it would overflow), -1 if undefined
float pull(vec4 other, vec2 at) {
    vec2 d = other.xy - at;
    float s = max(abs(d.x), abs(d.y));
    if (s <= 0.0 || other.z <= 0.0) return -1.0;
    vec2 n = d / s;
    return (other.z / s) / (s * dot(n, n));
}

void main()
{
    uint probe = gl_WorkGroupID.x;
    int lid = int(gl_LocalInvocationID.x);

    int target = probes[probe].info.x;
    bool valid = target >= 0 && target < u_count;
    vec4 self_pmr = valid ? bodies[target].pmr : vec4(0.0);

    //the strongest attractors this thread found
    float best_value[TOP_ATTRACTORS];
    int   best_index[TOP_ATTRACTORS];
    for (int k = 0; k < TOP_ATTRACTORS; ++k) {
        best_value[k] = -1.0;
        best_index[k] = -1;
    }
    float primary_value = -1.0;
    int   primary_index = -1;

    if (valid) {
        for (int j = lid; j < u_count; j += 256) {
            if (j == target) continue;

            //background bodies dont count as attractors
            if (is_background(bodies[j].meta.y)) continue;
            vec4 other = bodies[j].pmr;
            float value = pull(other, self_pmr.xy);
            if (value <= 0.0) continue;

            if (other.z > self_pmr.z && value > primary_value) {
                primary_value = value;
                primary_index = j;
            }

            if (value > best_value[TOP_ATTRACTORS - 1]) {
                best_value[TOP_ATTRACTORS - 1] = value;
                best_index[TOP_ATTRACTORS - 1] = j;
                for (int q = TOP_ATTRACTORS - 1; q > 0; --q) {
                    if (best_value[q] <= best_value[q - 1]) break;
                    float tv = best_value[q]; best_value[q] = best_value[q - 1]; best_value[q - 1] = tv;
                    int ti = best_index[q]; best_index[q] = best_index[q - 1]; best_index[q - 1] = ti;
                }
            }
        }
    }

    //finds the strongest heavier attractor of all threads
    s_value[lid] = primary_value;
    s_index[lid] = primary_index;
    barrier();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride && s_value[lid + stride] > s_value[lid]) {
            s_value[lid] = s_value[lid + stride];
            s_index[lid] = s_index[lid + stride];
        }
        barrier();
    }

    int strongest_heavier = s_index[0];
    barrier();

    //finds the strongest attractors of all threads, one per round
    int head = 0;
    int top[TOP_ATTRACTORS];
    for (int k = 0; k < TOP_ATTRACTORS; ++k) top[k] = -1;

    for (int r = 0; r < TOP_ATTRACTORS; ++r) {
        s_value[lid] = head < TOP_ATTRACTORS ? best_value[head] : -1.0;
        s_index[lid] = head < TOP_ATTRACTORS ? best_index[head] : -1;
        s_owner[lid] = lid;
        barrier();

        for (int stride = 128; stride > 0; stride >>= 1) {
            if (lid < stride && s_value[lid + stride] > s_value[lid]) {
                s_value[lid] = s_value[lid + stride];
                s_index[lid] = s_index[lid + stride];
                s_owner[lid] = s_owner[lid + stride];
            }
            barrier();
        }

        if (s_value[0] > 0.0) top[r] = s_index[0];
        if (s_owner[0] == lid) head++;
        barrier();
    }

    //finds the body the target orbits, which is not always the strongest heavier attractor
    //(the sun pulls harder on the moon than the earth does): of the heavier bodies among the four strongest
    //attractors it takes the one with the smallest hill sphere that contains the target
    int primary = -1;
    float primary_hill = 0.0;

    for (int c = 0; c < 4; ++c) {
        int candidate = top[c];
        if (candidate < 0) break;

        vec4 candidate_pmr = bodies[candidate].pmr;
        if (candidate_pmr.z <= self_pmr.z) continue;

        float parent_value = -1.0;
        int parent_index = -1;
        for (int j = lid; j < u_count; j += 256) {
            if (j == candidate) continue;
            vec4 other = bodies[j].pmr;
            if (other.z <= candidate_pmr.z || is_background(bodies[j].meta.y)) continue;

            float value = pull(other, candidate_pmr.xy);
            if (value > parent_value) {
                parent_value = value;
                parent_index = j;
            }
        }

        s_value[lid] = parent_value;
        s_index[lid] = parent_index;
        barrier();

        for (int stride = 128; stride > 0; stride >>= 1) {
            if (lid < stride && s_value[lid + stride] > s_value[lid]) {
                s_value[lid] = s_value[lid + stride];
                s_index[lid] = s_index[lid + stride];
            }
            barrier();
        }

        int parent = s_index[0];
        barrier();

        float hill = 3.0e38;
        if (parent >= 0) {
            vec4 parent_pmr = bodies[parent].pmr;
            hill = safe_length(candidate_pmr.xy - parent_pmr.xy) * pow(candidate_pmr.z / parent_pmr.z / 3.0, 1.0 / 3.0);
        }

        if (safe_length(self_pmr.xy - candidate_pmr.xy) < hill && (primary < 0 || hill < primary_hill)) {
            primary = candidate;
            primary_hill = hill;
        }
    }

    if (primary < 0) primary = strongest_heavier;

    if (lid == 0) {
        for (int k = 0; k < TOP_ATTRACTORS; ++k) probes[probe].top[k] = top[k];
        probes[probe].extra.x = primary;
    }
}
"""


#hill sphere and habitable zone as circles on screen (the radius are calculated every frame from the body data)
ZONE_VERTEX_SHADER = VERTEX_START + PROBE_STRUCT + STAR_FORMULAS + """
layout(std430, binding = 5) readonly buffer ProbeBuffer { Probe probes[]; };

struct Zone {
    ivec4 info;     //x = kind (0 = hill sphere, 1 = habitable zone), y = body, w = probe slot of the body or -1
    vec4  color;    //rgb, fill opacity
};

layout(std430, binding = 8) readonly buffer ZoneBuffer { Zone zones[]; };

flat out vec2 v_center;     //screen pixels
flat out vec2 v_radii;      //inner and outer radius in pixels
flat out vec4 v_color;

void main() {
    Zone zone = zones[gl_InstanceID];
    int kind = zone.info.x;
    int body = zone.info.y;

    gl_Position = CULLED;
    v_center = vec2(0.0);
    v_radii = vec2(0.0);
    v_color = zone.color;

    if (!valid_body(body)) return;

    vec4 b = bodies[body].pmr;
    float inner = 0.0;
    float outer = 0.0;

    if (kind == 1) {
        //habitable zone around a star
        float l = luminosity(b.z);
        inner = sqrt(l / 1.1) * AU_IN_METERS;
        outer = sqrt(l / 0.53) * AU_IN_METERS;
    } else {
        //hill sphere of the body around what it orbits
        int primary = zone.info.w >= 0 ? probes[zone.info.w].extra.x : -1;
        if (!valid_body(primary) || primary == body) return;
        vec4 p = bodies[primary].pmr;
        if (p.z <= 0.0 || b.z <= 0.0) return;
        outer = safe_length(b.xy - p.xy) * pow(b.z / p.z / 3.0, 1.0 / 3.0);
    }

    vec2 c = body_to_screen(body, camera_center());
    float r_outer = outer * u_zoom;
    float r_inner = inner * u_zoom;

    if (!(r_outer > 1.0)) return;

    //bounds of the circle, clipped to the screen
    vec2 lo = max(c - r_outer - 2.0, vec2(-2.0));
    vec2 hi = min(c + r_outer + 2.0, u_resolution + 2.0);
    if (lo.x >= hi.x || lo.y >= hi.y) return;

    vec2 corner = quad_corner() * 0.5 + 0.5;
    gl_Position = screen_to_clip(mix(lo, hi, corner));

    v_center = c;
    v_radii = vec2(r_inner, r_outer);
}
"""


ZONE_FRAGMENT_SHADER = FRAGMENT_START + """
flat in vec2 v_center;
flat in vec2 v_radii;
flat in vec4 v_color;

out vec4 fragColor;

void main() {
    vec2 frag = vec2(gl_FragCoord.x, u_resolution.y - gl_FragCoord.y);
    float d = distance(frag, v_center);

    float inner = v_radii.x;
    float outer = v_radii.y;

    float fill = clamp(outer - d + 0.5, 0.0, 1.0);
    if (inner > 0.0) fill *= clamp(d - inner + 0.5, 0.0, 1.0);

    float edge = clamp(1.25 - abs(d - outer), 0.0, 1.0);
    if (inner > 0.0) edge = max(edge, clamp(1.25 - abs(d - inner), 0.0, 1.0));

    float alpha = max(fill * v_color.a, edge * 0.85);
    if (alpha <= 0.003) discard;
    fragColor = vec4(v_color.rgb, alpha);
}
"""


#roche limits - a thin ring around the body causing the tides, shown when another body comes near its roche limit
ROCHE_RING_VERTEX_SHADER = VERTEX_START + AUX_BUFFER.replace("buffer AuxBuffer", "readonly buffer AuxBuffer") + ROCHE_FUNCTION + """
uniform float u_G;

flat out vec2 v_center;     //screen pixels
flat out float v_radius;    //pixels
flat out float v_fade;
flat out float v_inside;

void main() {
    int i = gl_InstanceID;
    gl_Position = CULLED;
    v_center = vec2(0.0);
    v_radius = 0.0;
    v_fade = 0.0;
    v_inside = 0.0;

    vec4 b = bodies[i].pmr;
    int source = aux[i].tide_source;
    if (int(bodies[i].meta.x + 0.5) == 2 || b.z <= 0.0 || b.w <= 0.0 || !valid_body(source)
            || aux[i].tidal <= 0.0) return;

    float d = sqrt(aux[i].tidal_r2);
    float roche = roche_distance(aux[i].tidal, d, b.w, u_G * b.z);
    float closeness = (d - b.w) / roche;    //distance of the near surface divided by the roche distance
    float fade = 1.0 - smoothstep(1.3, 2.0, closeness);
    float r = roche * u_zoom;
    if (fade <= 0.0 || r < 4.0) return;

    vec2 c = body_to_screen(source, camera_center());
    vec2 lo = max(c - r - 2.0, vec2(-2.0));
    vec2 hi = min(c + r + 2.0, u_resolution + 2.0);
    if (lo.x >= hi.x || lo.y >= hi.y) return;

    gl_Position = screen_to_clip(mix(lo, hi, quad_corner() * 0.5 + 0.5));
    v_center = c;
    v_radius = r;
    v_fade = fade;
    v_inside = closeness < 1.0 ? 1.0 : 0.0;
}
"""


ROCHE_RING_FRAGMENT_SHADER = FRAGMENT_START + """
flat in vec2 v_center;
flat in float v_radius;
flat in float v_fade;
flat in float v_inside;

out vec4 fragColor;

const vec3 ROCHE_COLOR = vec3(0.92, 0.47, 0.40);

void main() {
    vec2 frag = vec2(gl_FragCoord.x, u_resolution.y - gl_FragCoord.y);
    float d = distance(frag, v_center);

    float ring = clamp(1.0 - abs(d - v_radius), 0.0, 1.0);
    float fill = clamp(v_radius - d + 0.5, 0.0, 1.0) * 0.08 * v_inside;

    float alpha = max(ring * 0.65, fill) * v_fade;
    if (alpha <= 0.003) discard;
    fragColor = vec4(ROCHE_COLOR, alpha);
}
"""


#shared by the arrow shaders - finds the two bodies of an arrow and calculates where it starts and how long it is
ARROW_FUNCTIONS = PROBE_STRUCT + """
layout(std430, binding = 5) readonly buffer ProbeBuffer { Probe probes[]; };

struct Arrow {
    ivec4 info;    //x = target, y = its probe slot, z = rank among its strongest attractors
};

layout(std430, binding = 9) readonly buffer ArrowBuffer { Arrow arrows[]; };

uniform int   u_arrow_count;
uniform float u_log10_G;
uniform vec4  u_arrow_style;   //x = min length, y = max length, z = shaft width, w = gap to body (px)

const float LOG10_2 = 0.30102999566;

int arrow_source(int a) {
    int slot = arrows[a].info.y;
    return slot >= 0 ? probes[slot].top[clamp(arrows[a].info.z, 0, TOP_ATTRACTORS - 1)] : -1;
}

//log10 of the force (N) between source and target (-1e30 if invalid)
//calculated with logarithms because the force overflows a float
float arrow_log_force(int target, int source) {
    if (!valid_body(target) || !valid_body(source) || source == target) return -1e30;

    vec4 t = bodies[target].pmr;
    vec4 s = bodies[source].pmr;
    float r = safe_length(s.xy - t.xy);
    if (t.z <= 0.0 || s.z <= 0.0 || r <= 0.0) return -1e30;

    return u_log10_G + (log2(t.z) + log2(s.z) - 2.0 * log2(r)) * LOG10_2;
}

vec2 arrow_direction(int target, int source) {
    vec2 d = bodies[source].pmr.xy - bodies[target].pmr.xy;
    return normalize(d / max(abs(d.x), abs(d.y)));
}

struct ArrowShape {
    bool  ok;
    vec2  start;       //screen pixels
    vec2  dir;         //direction on screen
    float len;         //pixels
    float log_force;
    vec3  color;
};

ArrowShape arrow_shape(int a, vec2 cam) {
    ArrowShape shape;
    shape.ok = false;
    shape.start = vec2(0.0);
    shape.dir = vec2(1.0, 0.0);
    shape.len = 0.0;
    shape.log_force = 0.0;
    shape.color = vec3(0.0);

    int target = arrows[a].info.x;
    int source = arrow_source(a);
    float log_force = arrow_log_force(target, source);
    if (log_force < -1e29) return shape;

    //the length shows the strength compared to the strongest arrow (over four orders of magnitude)
    float strongest = log_force;
    for (int other = 0; other < u_arrow_count; ++other)
        strongest = max(strongest, arrow_log_force(arrows[other].info.x, arrow_source(other)));
    float strength = clamp(1.0 + (log_force - strongest) / 4.0, 0.0, 1.0);
    float len = mix(u_arrow_style.x, u_arrow_style.y, strength);

    Body t = bodies[target];
    Body s = bodies[source];
    float body_px = visual_radius_px(t);

    //the arrow is only drawn while it ends before the edge of the source body
    float distance_px = safe_length(s.pmr.xy - t.pmr.xy) * u_zoom;
    if (body_px + u_arrow_style.w + len >= distance_px - visual_radius_px(s)) return shape;

    shape.ok = true;
    shape.dir = arrow_direction(target, source);
    shape.len = len;
    shape.start = body_to_screen(target, cam) + shape.dir * (body_px + u_arrow_style.w);
    shape.log_force = log_force;
    //pale version of the color of the source body
    vec3 source_color = bodies[source].cd.rgb;
    float grey = dot(source_color, vec3(0.299, 0.587, 0.114));
    shape.color = mix(vec3(0.84, 0.88, 0.95), mix(vec3(grey), source_color, 0.6), 0.35);

    return shape;
}
"""


#ids of the characters the label shaders use
GLYPH_IDS = f"""const int GLYPH_E = {GLYPHS.index("e")};
const int GLYPH_MINUS = {GLYPHS.index("-")};
const int GLYPH_N = {GLYPHS.index("N")};
const int GLYPH_SPACE = {GLYPHS.index(" ")};
const int GLYPH_COMMA = {GLYPHS.index(",")};
const int GLYPH_L = {GLYPHS.index("L")};
const int GLYPH_S = {GLYPHS.index("S")};
"""

LABEL_FUNCTIONS = GLYPH_IDS + """
//returns character i of a label like "3,5e22 N" (-1 after the end)
int label_glyph(int i, float log_value, out int char_count) {
    int e = int(floor(log_value));
    int m = int(floor(pow(10.0, log_value - float(e)) * 10.0 + 0.5));
    if (m >= 100) { m = 10; e += 1; }
    m = max(m, 10);

    int ae = abs(e);
    int digits = ae >= 100 ? 3 : (ae >= 10 ? 2 : 1);
    int sign = e < 0 ? 1 : 0;
    char_count = 4 + sign + digits + 2;

    if (i == 0) return m / 10;
    if (i == 1) return GLYPH_COMMA;
    if (i == 2) return m % 10;
    if (i == 3) return GLYPH_E;

    int k = i - 4;
    if (sign == 1) {
        if (k == 0) return GLYPH_MINUS;
        k -= 1;
    }
    if (k < digits) {
        int power = digits - 1 - k;
        int divisor = power == 2 ? 100 : (power == 1 ? 10 : 1);
        return (ae / divisor) % 10;
    }
    if (k == digits) return GLYPH_SPACE;
    if (k == digits + 1) return GLYPH_N;
    return -1;
}

//labels are dimmed until the mouse comes near
uniform vec2 u_mouse;
float label_fade(vec2 center) {
    return mix(0.45, 1.0, 1.0 - smoothstep(30.0, 110.0, distance(center, u_mouse)));
}
"""


MARKER_STRUCTS = """
struct Marker {
    ivec4 info;    //x = kind (0 = barycenter, 1-5 = lagrange point L1-L5), y = the body, w = its probe slot
    vec4  color;
};

struct MarkerLayout {
    vec4 place;    //xy = screen position (px), z = visible, w = kind
    vec4 color;
};
"""


ARROW_LAYOUT_STRUCT = """
struct ArrowLayout {
    vec4 placement;   //xy = start, zw = direction (screen pixels)
    vec4 shape;       //x = length (px), y = log10 of the force (N), z = visible
    vec4 label;       //xy = label center, zw = label size (px)
    vec4 color;       //rgb
};
"""


#calculates position and label of every arrow once per frame (labels that would overlap get moved)
ARROW_LAYOUT = VERSION + BODIES_READ_ONLY + CAMERA_BLOCK + CAMERA_FUNCTIONS + ARROW_FUNCTIONS + LABEL_FUNCTIONS + ARROW_LAYOUT_STRUCT + MARKER_STRUCTS + """
layout(local_size_x = 256) in;

layout(std430, binding = 11) writeonly buffer ArrowLayoutBuffer { ArrowLayout layouts[]; };
layout(std430, binding = 15) readonly buffer MarkerLayoutBuffer { MarkerLayout marker_layouts[]; };

uniform vec2  u_glyph_size;      //size of one character cell on screen (px)
uniform float u_glyph_advance;   //distance between characters on screen (px)
uniform int   u_marker_count;

shared vec4  s_label[256];     //xy = center, zw = size
shared float s_push[256];      //direction the label moves in to get out of the way
shared int   s_visible[256];

//returns the box of the label of a marker ("S" or "L1")
bool marker_label(int m, out vec4 label) {
    MarkerLayout marker = marker_layouts[m];
    label = vec4(0.0);
    if (marker.place.z < 0.5) return false;

    float chars = int(marker.place.w + 0.5) == 0 ? 1.0 : 2.0;
    vec2 box = vec2(chars * u_glyph_advance, u_glyph_size.y);
    vec2 origin = floor(marker.place.xy + vec2(6.0, -4.0 - u_glyph_size.y) + 0.5);
    label = vec4(origin + box * 0.5, box);
    return true;
}

void main()
{
    int i = int(gl_LocalInvocationID.x);
    int count = min(u_arrow_count, 256);

    ArrowShape shape;
    shape.ok = false;

    s_label[i] = vec4(0.0);
    s_push[i] = 1.0;
    s_visible[i] = 0;

    if (i < count) {
        shape = arrow_shape(i, camera_center());

        if (shape.ok) {
            int char_count;
            label_glyph(0, shape.log_force, char_count);

            //the label sits a bit beyond the arrow tip so it never covers its own arrow
            vec2 box = vec2(float(char_count) * u_glyph_advance, u_glyph_size.y);
            vec2 tip = shape.start + shape.dir * shape.len;
            vec2 center = tip + shape.dir * (abs(shape.dir.x) * box.x * 0.5 + abs(shape.dir.y) * box.y * 0.5 + 4.0);

            s_label[i] = vec4(center, box);
            s_push[i] = shape.dir.y >= 0.0 ? 1.0 : -1.0;
            s_visible[i] = 1;
        }
    }
    barrier();

    //moves every label up or down until it doesnt overlap a marker label or an earlier arrow label
    //(done by one thread)
    if (i == 0) {
        for (int k = 0; k < count; ++k) {
            if (s_visible[k] == 0) continue;

            vec4 label = s_label[k];
            for (int pass = 0; pass < 2; ++pass) {
                for (int m = 0; m < u_marker_count; ++m) {
                    vec4 other;
                    if (!marker_label(m, other)) continue;
                    vec2 gap = abs(label.xy - other.xy) - (label.zw + other.zw) * 0.5;
                    if (gap.x < 1.0 && gap.y < 1.0)
                        label.y = other.y + s_push[k] * ((label.w + other.w) * 0.5 + 1.0);
                }
                for (int j = 0; j < k; ++j) {
                    if (s_visible[j] == 0) continue;
                    vec2 gap = abs(label.xy - s_label[j].xy) - (label.zw + s_label[j].zw) * 0.5;
                    if (gap.x < 1.0 && gap.y < 1.0)
                        label.y = s_label[j].y + s_push[k] * ((label.w + s_label[j].w) * 0.5 + 1.0);
                }
            }
            s_label[k] = label;
        }
    }
    barrier();

    if (i < count) {
        layouts[i].placement = vec4(shape.start, shape.dir);
        layouts[i].shape = vec4(shape.len, shape.log_force, float(s_visible[i]), 0.0);
        layouts[i].label = s_label[i];
        layouts[i].color = vec4(shape.color, 1.0);
    }
}
"""


ARROW_VERTEX_SHADER = VERTEX_START + ARROW_LAYOUT_STRUCT + """
layout(std430, binding = 11) readonly buffer ArrowLayoutBuffer { ArrowLayout layouts[]; };

uniform vec4 u_arrow_style;  //x = min length, y = max length, z = shaft width, w = gap to body (px)
uniform vec2 u_arrow_head;   //head length, half head width (px)

out vec2 v_local;            //position along / across the arrow in pixels
flat out float v_length;
flat out vec3 v_color;

void main() {
    ArrowLayout arrow = layouts[gl_InstanceID];
    vec2 start = arrow.placement.xy;
    vec2 dir = arrow.placement.zw;
    float len = arrow.shape.x;

    v_local = vec2(0.0);
    v_length = len;
    v_color = arrow.color.rgb;

    if (arrow.shape.z < 0.5) {
        gl_Position = CULLED;
        return;
    }

    float half_width = max(u_arrow_head.y, u_arrow_style.z * 0.5) + 1.5;
    vec2 corner = quad_corner();
    float along = mix(-1.5, len + 1.5, corner.x * 0.5 + 0.5);
    float across = corner.y * half_width;
    vec2 normal = vec2(-dir.y, dir.x);

    gl_Position = screen_to_clip(start + dir * along + normal * across);
    v_local = vec2(along, across);
}
"""


ARROW_FRAGMENT_SHADER = FRAGMENT_START + """
uniform vec4 u_arrow_style;
uniform vec2 u_arrow_head;

in vec2 v_local;
flat in float v_length;
flat in vec3 v_color;

out vec4 fragColor;

void main() {
    float u = v_local.x;
    float v = abs(v_local.y);

    float head_length = min(u_arrow_head.x, v_length * 0.6);
    float shaft_end = v_length - head_length;

    //distance in pixels to the shaft and to the arrow head
    float d_shaft = max(max(-u, u - shaft_end), v - u_arrow_style.z * 0.5);
    float slope = u_arrow_head.y / head_length;
    float d_side = (v - slope * (v_length - u)) / sqrt(1.0 + slope * slope);
    float d_head = max(shaft_end - u, d_side);

    float alpha = clamp(0.5 - min(d_shaft, d_head), 0.0, 1.0);
    if (alpha <= 0.003) discard;
    fragColor = vec4(v_color, alpha * 0.7);
}
"""


#force labels
LABEL_VERTEX_SHADER = VERTEX_START + ARROW_LAYOUT_STRUCT + LABEL_FUNCTIONS + """
layout(std430, binding = 11) readonly buffer ArrowLayoutBuffer { ArrowLayout layouts[]; };

uniform vec2  u_glyph_size;      //size of one character cell on screen (px)
uniform float u_glyph_advance;   //distance between characters on screen (px)

const int LABEL_CHARS = __LABEL_CHARS__;

out vec2 v_uv;
flat out int v_glyph;
flat out vec3 v_color;
flat out float v_fade;

void main() {
    int a = gl_InstanceID / LABEL_CHARS;
    int c = gl_InstanceID % LABEL_CHARS;
    ArrowLayout arrow = layouts[a];

    v_uv = vec2(0.0);
    v_glyph = 0;
    v_color = arrow.color.rgb;
    v_fade = label_fade(arrow.label.xy);

    int char_count;
    int glyph = label_glyph(c, arrow.shape.y, char_count);
    if (arrow.shape.z < 0.5 || glyph < 0 || glyph == GLYPH_SPACE) {
        gl_Position = CULLED;
        return;
    }

    //every character is centered in its slot
    vec2 center = arrow.label.xy;
    vec2 box = arrow.label.zw;
    vec2 origin = floor(center - box * 0.5 + 0.5);
    vec2 cell = origin + vec2(float(c) * u_glyph_advance + (u_glyph_advance - u_glyph_size.x) * 0.5, 0.0);
    vec2 corner = quad_corner() * 0.5 + 0.5;

    gl_Position = screen_to_clip(cell + corner * u_glyph_size);
    v_uv = corner;
    v_glyph = glyph;
}
""".replace("__LABEL_CHARS__", str(LABEL_CHARS))


LABEL_FRAGMENT_SHADER = FRAGMENT_START + """
uniform sampler2D u_glyphs;
uniform float u_glyph_count;
uniform vec2 u_shadow_offset;   //one screen pixel in texture coordinates

in vec2 v_uv;
flat in int v_glyph;
flat in vec3 v_color;
flat in float v_fade;

out vec4 fragColor;

void main() {
    vec2 uv = vec2((float(v_glyph) + v_uv.x) / u_glyph_count, v_uv.y);

    float ink = texture(u_glyphs, uv).r;
    float shadow = texture(u_glyphs, uv - u_shadow_offset).r * 0.55;

    //character over a soft shadow so the labels are readable on every background
    float alpha = ink + shadow * (1.0 - ink);
    if (alpha <= 0.003) discard;
    fragColor = vec4(v_color * ink / alpha, alpha * v_fade);
}
"""


### MARKERS (BARYCENTERS AND LAGRANGE POINTS) ###

MARKER_LAYOUT = VERSION + BODIES_READ_ONLY + CAMERA_BLOCK + CAMERA_FUNCTIONS + PROBE_STRUCT + MARKER_STRUCTS + """
layout(local_size_x = 256) in;

layout(std430, binding = 5) readonly buffer ProbeBuffer { Probe probes[]; };
layout(std430, binding = 13) readonly buffer MarkerBuffer { Marker markers[]; };
layout(std430, binding = 15) writeonly buffer MarkerLayoutBuffer { MarkerLayout marker_layouts[]; };

//pixels a lagrange marker stays away from its two bodies
const float CLEARANCE = 10.0;

shared vec4  s_sum[256];

//lagrange points on the line through both bodies, in units of their distance, for the mass ratio mu = m2 / (m1 + m2)
//L1 and L2 are measured from the lighter body, L3 from the heavier one (solved with Newton's method)
float lagrange_l1(float mu) {
    float r = clamp(pow(mu / 3.0, 1.0 / 3.0), 1e-7, 0.9);
    for (int i = 0; i < 16; ++i) {
        float a = 1.0 - r;
        float g = mu / (r * r) - r * (1.0 + (1.0 - mu) * (2.0 - r) / (a * a));
        float dg = -2.0 * mu / (r * r * r) - 1.0 - 2.0 * (1.0 - mu) / (a * a * a);
        r = clamp(r - g / dg, 1e-7, 0.999);
    }
    return r;
}

float lagrange_l2(float mu) {
    float r = clamp(pow(mu / 3.0, 1.0 / 3.0), 1e-7, 2.0);
    for (int i = 0; i < 16; ++i) {
        float a = 1.0 + r;
        float g = r * (1.0 + (1.0 - mu) * (2.0 + r) / (a * a)) - mu / (r * r);
        float dg = 1.0 + 2.0 * (1.0 - mu) / (a * a * a) + 2.0 * mu / (r * r * r);
        r = clamp(r - g / dg, 1e-7, 2.0);
    }
    return r;
}

float lagrange_l3(float mu) {
    float s = 1.0 - 7.0 * mu / 12.0;
    for (int i = 0; i < 16; ++i) {
        float b = 1.0 + s;
        float g = (1.0 - mu) / (s * s) + mu / (b * b) - mu - s;
        float dg = -2.0 * (1.0 - mu) / (s * s * s) - 2.0 * mu / (b * b * b) - 1.0;
        s = clamp(s - g / dg, 0.1, 2.0);
    }
    return s;
}

void main()
{
    uint m = gl_WorkGroupID.x;
    int lid = int(gl_LocalInvocationID.x);

    Marker marker = markers[m];
    int kind = marker.info.x;
    int slot = marker.info.w;
    vec2 cam = camera_center();

    if (kind == 0) {
        int body = marker.info.y;
        bool ok = valid_body(body) && slot >= 0 && bodies[body].pmr.z > 0.0;

        int primary = ok ? probes[slot].extra.x : -1;
        if (!valid_body(primary) || primary == body || bodies[primary].pmr.z <= 0.0) primary = -1;
        if (ok && primary < 0) {
            int strongest = probes[slot].top[0];
            if (valid_body(strongest) && strongest != body
                    && bodies[strongest].pmr.z >= COMPANION * bodies[body].pmr.z) primary = strongest;
        }
        ok = ok && primary >= 0;

        vec4 p = ok ? bodies[primary].pmr : vec4(0.0);
        vec2 primary_lo = ok ? bodies[primary].lo.xy : vec2(0.0);
        vec2 own = ok ? (bodies[body].pmr.xy - p.xy) + (bodies[body].lo.xy - primary_lo) : vec2(0.0);
        float reach = 0.5 * safe_length(own);

        //positions relative to the primary and masses in primary masses, so the sum stays exact
        vec3 sum = vec3(0.0);   //xy = sum of weight * position, z = sum of weights
        for (int j = lid; ok && j < u_body_count; j += 256) {
            vec4 other = bodies[j].pmr;
            if (other.z <= 0.0) continue;
            vec2 d = (other.xy - p.xy) + (bodies[j].lo.xy - primary_lo);
            if (j != body && j != primary && (is_background(bodies[j].meta.y) || safe_length(d) > reach)) continue;
            float weight = other.z / p.z;
            sum += vec3(weight * d, weight);
        }

        s_sum[lid] = vec4(sum, 0.0);
        barrier();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (lid < stride) s_sum[lid] += s_sum[lid + stride];
            barrier();
        }

        if (lid == 0) {
            vec4 total = s_sum[0];
            bool found = ok && total.z > 0.0;
            vec2 point = found ? total.xy / total.z : vec2(0.0);

            //the marker is only drawn while the body is clearly away from it on screen
            bool visible = found && safe_length(own - point) * u_zoom > CLEARANCE + 5.0;

            marker_layouts[m].place = vec4(found ? body_to_screen(primary, cam) + point * u_zoom : vec2(0.0),
                                           visible ? 1.0 : 0.0, 0.0);
            marker_layouts[m].color = marker.color;
        }
        return;
    }

    if (lid != 0) return;

    marker_layouts[m].place = vec4(0.0, 0.0, 0.0, float(kind));
    marker_layouts[m].color = marker.color;

    int secondary = marker.info.y;
    int primary = slot >= 0 ? probes[slot].extra.x : -1;
    if (!valid_body(primary) || !valid_body(secondary) || primary == secondary) return;

    //the heavier one of the two is the primary
    Body p = bodies[primary];
    Body s = bodies[secondary];
    if (s.pmr.z > p.pmr.z) {
        Body t = p;
        p = s;
        s = t;
    }
    if (s.pmr.z <= 0.0) return;

    vec2 rel = (s.pmr.xy - p.pmr.xy) + (s.lo.xy - p.lo.xy);
    float separation = safe_length(rel);
    if (separation <= 0.0) return;

    vec2 axis = rel / separation;
    float mu = 1.0 / (1.0 + p.pmr.z / s.pmr.z);

    //L4 is ahead of the lighter body on its orbit, L5 behind it
    vec2 relative_velocity = s.va.xy - p.va.xy;
    float spin = axis.x * relative_velocity.y - axis.y * relative_velocity.x;
    vec2 ahead = (spin >= 0.0 ? 1.0 : -1.0) * vec2(-axis.y, axis.x);

    vec2 point;   //from the primary, in units of the distance
    if (kind == 1) point = axis * (1.0 - lagrange_l1(mu));
    else if (kind == 2) point = axis * (1.0 + lagrange_l2(mu));
    else if (kind == 3) point = -axis * lagrange_l3(mu);
    else if (kind == 4) point = axis * 0.5 + ahead * 0.8660254;
    else point = axis * 0.5 - ahead * 0.8660254;

    vec2 primary_px = world_to_screen(p.pmr.xy, cam) + p.lo.xy * u_zoom;
    vec2 secondary_px = world_to_screen(s.pmr.xy, cam) + s.lo.xy * u_zoom;
    vec2 screen = primary_px + point * (separation * u_zoom);

    //only drawn while the point is clearly away from both bodies on screen
    bool visible = distance(screen, secondary_px) > visual_radius_px(s) + CLEARANCE
                && distance(screen, primary_px) > visual_radius_px(p) + CLEARANCE;

    marker_layouts[m].place = vec4(screen, visible ? 1.0 : 0.0, float(kind));
}
"""


MARKER_VERTEX_SHADER = VERTEX_START + MARKER_STRUCTS + """
layout(std430, binding = 15) readonly buffer MarkerLayoutBuffer { MarkerLayout marker_layouts[]; };

const float MARKER_HALF_SIZE = 7.0;

out vec2 v_local;        //pixels from the center of the marker
flat out int v_kind;
flat out vec4 v_color;

void main() {
    MarkerLayout marker = marker_layouts[gl_InstanceID];

    v_local = vec2(0.0);
    v_kind = int(marker.place.w + 0.5);
    v_color = marker.color;

    if (marker.place.z < 0.5) {
        gl_Position = CULLED;
        return;
    }

    vec2 corner = quad_corner() * MARKER_HALF_SIZE;
    gl_Position = screen_to_clip(marker.place.xy + corner);
    v_local = corner;
}
"""


MARKER_FRAGMENT_SHADER = FRAGMENT_START + """
in vec2 v_local;
flat in int v_kind;
flat in vec4 v_color;

out vec4 fragColor;

//shape of the marker: barycenter = ring with two filled quarters, lagrange point = small thin ring
//grow makes it bigger for the dark outline
float symbol(vec2 p, float grow) {
    float r = length(p);
    if (v_kind == 0) {
        float ring = clamp(0.9 + grow - abs(r - 4.5), 0.0, 1.0);
        float quarters = p.x * p.y > 0.0 ? clamp(4.6 + grow - r, 0.0, 1.0) : 0.0;
        return max(ring, quarters);
    }
    return clamp(0.75 + grow - abs(r - 3.5), 0.0, 1.0);
}

void main() {
    float ink = symbol(v_local, 0.0);
    float halo = symbol(v_local, 1.2) * 0.5;

    float alpha = ink + halo * (1.0 - ink);
    if (alpha <= 0.003) discard;
    fragColor = vec4(v_color.rgb * ink / alpha, alpha * v_color.a);
}
"""


#labels of the markers: "S" (Schwerpunkt) for barycenters and "L1" to "L5" for lagrange points
MARKER_LABEL_VERTEX_SHADER = VERTEX_START + MARKER_STRUCTS + GLYPH_IDS + """
layout(std430, binding = 15) readonly buffer MarkerLayoutBuffer { MarkerLayout marker_layouts[]; };

uniform vec2  u_glyph_size;      //size of one character cell on screen (px)
uniform float u_glyph_advance;   //distance between characters on screen (px)

//dimmed until the mouse comes near
uniform vec2 u_mouse;

out vec2 v_uv;
flat out int v_glyph;
flat out vec3 v_color;
flat out float v_fade;

void main() {
    int m = gl_InstanceID / 2;
    int c = gl_InstanceID % 2;
    MarkerLayout marker = marker_layouts[m];
    int kind = int(marker.place.w + 0.5);

    v_uv = vec2(0.0);
    //"S" for a barycenter, "L" and its number for a lagrange point
    v_glyph = c == 0 ? (kind == 0 ? GLYPH_S : GLYPH_L) : kind;
    v_color = marker.color.rgb;
    v_fade = mix(0.4, 1.0, 1.0 - smoothstep(30.0, 110.0, distance(marker.place.xy, u_mouse)));

    if (marker.place.z < 0.5 || (kind == 0 && c == 1)) {
        gl_Position = CULLED;
        return;
    }

    //above and to the right of the marker
    vec2 origin = floor(marker.place.xy + vec2(6.0, -4.0 - u_glyph_size.y) + 0.5);
    vec2 cell = origin + vec2(float(c) * u_glyph_advance + (u_glyph_advance - u_glyph_size.x) * 0.5, 0.0);
    vec2 corner = quad_corner() * 0.5 + 0.5;

    gl_Position = screen_to_clip(cell + corner * u_glyph_size);
    v_uv = corner;
}
"""


#copies everything the inspect window shows about a body into a small buffer the cpu reads a few frames later
INSPECT_SAMPLE = VERSION + BODIES_READ_ONLY + PROBE_STRUCT + """
layout(local_size_x = 256) in;

layout(std430, binding = 5) readonly buffer ProbeBuffer { Probe probes[]; };

layout(std430, binding = 16) writeonly buffer InspectSampleBuffer {
    Body  sample_bodies[6];   //the body, what it orbits and its four strongest attractors
    ivec4 sample_info;        //x = index of the primary or -1, y = other bodies with at least 1 % of its mass,
                              //z = highest index of those, w = 1 if the primary is a lighter partner
    ivec4 sample_top;         //indices of the four strongest attractors (-1 = none)
    vec4  sample_system[2];   //relative to the primary, masses in primary masses:
                              //[0] = total mass, mass-weighted position.xy
                              //[1] = mass-weighted velocity.xy
};

uniform int u_target;
uniform int u_probe;
uniform int u_count;

const float SIGNIFICANT = 0.01;

shared vec4 s_position[256];
shared vec4 s_velocity[256];
shared int s_members;
shared int s_partner;

void main()
{
    int lid = int(gl_LocalInvocationID.x);

    int primary = u_probe >= 0 ? probes[u_probe].extra.x : -1;
    if (primary < 0 || primary >= u_count || primary == u_target) primary = -1;

    int companion = 0;
    if (primary < 0 && u_probe >= 0) {
        int strongest = probes[u_probe].top[0];
        if (strongest >= 0 && strongest < u_count && strongest != u_target &&
            bodies[strongest].pmr.z >= COMPANION * bodies[u_target].pmr.z) {
            primary = strongest;
            companion = 1;
        }
    }

    vec4 position_sum = vec4(0.0);
    vec4 velocity_sum = vec4(0.0);

    if (lid == 0) { s_members = 0; s_partner = -1; }
    barrier();

    if (primary >= 0) {
        vec4 p = bodies[primary].pmr;
        vec2 pv = bodies[primary].va.xy;
        float reach = 0.5 * safe_length(bodies[u_target].pmr.xy - p.xy);

        for (int j = lid; j < u_count; j += 256) {
            if (j == u_target || is_background(bodies[j].meta.y)) continue;
            vec4 other = bodies[j].pmr;
            vec2 d = other.xy - p.xy;
            if (safe_length(d) > reach) continue;

            float w = other.z / p.z;
            position_sum += vec4(w, w * d, 0.0);
            velocity_sum += vec4(w * (bodies[j].va.xy - pv), 0.0, 0.0);

            if (j != primary && w >= SIGNIFICANT) {
                atomicAdd(s_members, 1);
                atomicMax(s_partner, j);
            }
        }
    }

    s_position[lid] = position_sum;
    s_velocity[lid] = velocity_sum;
    barrier();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) {
            s_position[lid] += s_position[lid + stride];
            s_velocity[lid] += s_velocity[lid + stride];
        }
        barrier();
    }

    if (lid == 0) {
        sample_bodies[0] = bodies[u_target];
        sample_bodies[1] = bodies[primary >= 0 ? primary : u_target];
        sample_info = ivec4(primary, s_members, s_partner, companion);

        ivec4 top = ivec4(-1);
        for (int k = 0; k < 4; ++k) {
            if (u_probe >= 0) top[k] = probes[u_probe].top[k];
            if (top[k] >= u_count || top[k] == u_target) top[k] = -1;
            sample_bodies[2 + k] = bodies[top[k] >= 0 ? top[k] : u_target];
        }
        sample_top = top;
        sample_system[0] = s_position[0];
        sample_system[1] = s_velocity[0];
    }
}
"""


### STABILITY MAP (stability_map.py) ###

MAP_BODIES = 32          #max bodies the map takes into account
MAP_ROWS = 64           #start speeds (one invocation each)
MAP_STEPS = 5760        #max steps of the bodies per column

#shared by the two map shaders: the bodies of the map, the data of a column and the two-body orbit solver
MAP_COMMON = """
const int   BODIES = __BODIES__;
const int   MAX_STEPS = __STEPS__;
const float BODY_STEPS = 8.0;       //min steps per orbit of the fastest body

struct MapBody {
    vec4 own;       //position.xy, velocity.xy relative to the center of mass of S and all nearer bodies
    vec4 centre;    //that point relative to S: position.xy, velocity.xy
    vec4 info;      //x = mass / M_S, y = radius, z = from how far out it counts as part of the center,
                    //w = mass of S and all nearer bodies
    vec4 more;      //x = orbit period, y = its share of the mass (S + nearer bodies + itself),
                    //z = how close a particle has to come so it counts on its own again,
                    //w = how far inside its orbit a particle can be before it gets left out
};
layout(std430, binding = 19) readonly buffer MapSystem { MapBody map_bodies[]; };
layout(std430, binding = 14) buffer MapPaths { vec4 map_paths[]; };

uniform int   u_body_count;
uniform int   u_columns;
uniform int   u_first_column;     //column of the first workgroup
uniform float u_r_min;            //inner edge of the map (the outer edge is 1)
uniform float u_orbits;           //how many orbits of the start circle get followed
uniform float u_steps_per_orbit;

//data of a column: its start circle, how many of the nearest bodies count as part of the center,
//the mass of that center and its steps
struct Column {
    float r0;
    int   inner;
    float mu;
    float dt;
    int   steps;
};

Column column_of(int column) {
    Column c;
    int count = min(u_body_count, BODIES);
    c.r0 = u_r_min * pow(1.0 / u_r_min, (float(column) + 0.5) / float(u_columns));
    c.inner = 0;
    while (c.inner < count && map_bodies[c.inner].info.z < c.r0) c.inner++;
    c.mu = map_bodies[c.inner].info.w;
    float period = 6.2831853 * sqrt(c.r0 * c.r0 * c.r0 / c.mu);
    float dt = period / u_steps_per_orbit;
    for (int k = c.inner; k < count; ++k) dt = min(dt, map_bodies[k].more.x / BODY_STEPS);
    float span = u_orbits * period;
    c.steps = int(min(ceil(span / dt), float(MAX_STEPS)));
    c.dt = span / float(c.steps);
    return c;
}

int path_index(int group, int step, int body) {
    return (group * (MAX_STEPS + 1) + step) * BODIES + body;
}

//Stumpff functions c1, c2, c3 (needed for the universal Kepler solver)
void stumpff(float z, out float c1, out float c2, out float c3) {
    if (abs(z) < 0.25) {
        c2 = 0.5 - z * (1.0 / 24.0 - z * (1.0 / 720.0 - z * (1.0 / 40320.0)));
        c3 = 1.0 / 6.0 - z * (1.0 / 120.0 - z * (1.0 / 5040.0 - z * (1.0 / 362880.0)));
        c1 = 1.0 - z * c3;
    } else if (z > 0.0) {
        float x = sqrt(z);
        float half_sine = sin(0.5 * x);
        float sine = sin(x);
        c1 = sine / x;
        c2 = 2.0 * half_sine * half_sine / z;
        c3 = (x - sine) / (x * z);
    } else {
        float x = min(sqrt(-z), 40.0);
        float half_sine = sinh(0.5 * x);
        float sine = sinh(x);
        c1 = sine / x;
        c2 = 2.0 * half_sine * half_sine / (x * x);
        c3 = (sine - x) / (x * x * x);
    }
}

//moves (r, v) along its two-body orbit around G M = mu for dt
//works for all orbits (closed and open) and any step length (universal variables, solved with Newton's method)
void kepler_flow(inout vec2 r, inout vec2 v, float mu, float dt)
{
    float r0 = length(r);
    if (!(r0 > 0.0) || !(dt > 0.0)) return;
    float h = r.x * v.y - r.y * v.x;
    float beta = 2.0 * mu / r0 - dot(v, v);     //mu / a (above 0 for a closed orbit)
    float eta = dot(r, v);
    float zeta = mu - beta * r0;

    float low = 0.0, high;
    if (beta > 0.0) {
        float root = sqrt(beta);
        float period = 6.2831853 * mu / (beta * root);
        if (dt >= period) dt = mod(dt, period);     //full turns change nothing
        high = 6.2831853 / root;
    } else {
        //upper bound for s on an open orbit
        float e = sqrt(max(0.0, 1.0 - beta * h * h / (mu * mu)));
        high = dt * (1.0 + e) * mu / max(h * h, 1e-30);
        if (beta < 0.0) high = min(high, 20.0 / sqrt(-beta));
    }

    //start value for a short step
    float s = dt / r0;
    s = clamp(s * (1.0 - clamp(0.5 * eta * s / r0, -0.5, 0.5)), low, high);
    float c1, c2, c3;
    bool settled = false;
    for (int k = 0; k < 16; ++k) {
        stumpff(beta * s * s, c1, c2, c3);
        float value = s * (r0 + s * (eta * c2 + s * zeta * c3)) - dt;
        if (abs(value) <= 2.0e-7 * dt) { settled = true; break; }
        float slope = r0 + s * (eta * c1 + s * zeta * c2);
        if (value > 0.0) high = s; else low = s;
        float next = s - value / slope;
        s = (next >= low && next <= high) ? next : 0.5 * (low + high);
    }
    if (!settled) stumpff(beta * s * s, c1, c2, c3);

    float g1 = s * c1, g2 = s * s * c2, g3 = s * s * s * c3;
    float r1 = r0 + eta * g1 + zeta * g2;
    vec2 r_new = r - (mu * g2 / r0) * r + (dt - mu * g3) * v;
    v += -(mu * g1 / (r1 * r0)) * r - (mu * g2 / r1) * v;
    r = r_new;

    //corrects the sideways speed to keep the angular momentum exactly (see kepler_step)
    v += (h - (r.x * v.y - r.y * v.x)) / dot(r, r) * vec2(-r.y, r.x);
}
"""


STABILITY_BODIES = (VERSION + MAP_COMMON + """
layout(local_size_x = __BODIES__) in;

shared vec4 s_state[2 * BODIES];    //all bodies before and after the step (swapped every step)

void main()
{
    int group = int(gl_WorkGroupID.x);
    int k = int(gl_LocalInvocationID.x);
    int count = min(u_body_count, BODIES);
    Column c = column_of(u_first_column + group);

    bool mine = k < count;
    bool outer = k >= c.inner;
    float mu = 1.0;
    if (mine) {
        MapBody body = map_bodies[k];
        vec4 state = body.own;
        if (outer) state += body.centre - map_bodies[c.inner].centre;
        s_state[k] = state;
        mu = (outer ? c.mu : body.info.w) + body.info.x;
    }
    barrier();

    int from = 0;
    for (int step = 0; step < c.steps; ++step) {
        if (mine) {
            vec4 state = s_state[from + k];
            if (outer) {
                vec2 kick = vec2(0.0);
                for (int j = c.inner; j < count; ++j) {
                    if (j == k) continue;
                    vec2 other = s_state[from + j].xy;
                    vec2 d = other - state.xy;
                    float reach = map_bodies[j].info.y + map_bodies[k].info.y;
                    float d2 = max(dot(d, d), reach * reach);
                    float o2 = dot(other, other);
                    kick += map_bodies[j].info.x * (d * inversesqrt(d2) / d2 - other * inversesqrt(o2) / o2);
                }
                state.zw += kick * c.dt;
            }
            map_paths[path_index(group, step, k)] = state;
            kepler_flow(state.xy, state.zw, mu, c.dt);
            s_state[BODIES - from + k] = state;
        }
        barrier();
        from = BODIES - from;
    }
    if (mine) map_paths[path_index(group, c.steps, k)] = s_state[from + k];
}
""").replace("__BODIES__", str(MAP_BODIES)).replace("__STEPS__", str(MAP_STEPS))


STABILITY_MAP = (VERSION + MAP_COMMON + """
layout(local_size_x = __ROWS__) in;

struct MapParticle {
    vec4  motion;   //position.xy, velocity.xy around its center
    vec4  seen;     //x, y = nearest and farthest distance from its center, z = how far it is into the step of the bodies
    ivec4 at;       //x = step of the bodies it is in, y = how many of the nearest bodies are part of its center,
                    //z = steps taken, w = 1 when it is done
};
layout(std430, binding = 21) writeonly buffer MapCells { vec4 map_cells[]; };
layout(std430, binding = 23) buffer MapParticles { MapParticle map_particles[]; };
layout(std430, binding = 24) buffer MapLeft { uint map_left[]; };

uniform float u_speed_max;        //top edge in circular speeds
uniform vec2  u_direction;        //direction the particles start in (seen from S)
uniform float u_sense;            //+1 anticlockwise, -1 clockwise
uniform float u_impact;           //radius of S (3 Schwarzschild radii for a black hole)
uniform float u_escape;           //a particle this far out has escaped
uniform int   u_extra_steps;      //extra steps a particle can use for close encounters
uniform int   u_turns;            //steps per particle in this run
uniform int   u_resume;           //0 = from the start, 1 = continue from the last run

const float RESOLVE = 8.0;        //steps per orbit of a body the particle is close to
const float ENCOUNTER = 0.3;      //a step covers at most this share of the distance to a body ...
const float FALL = 0.1;           //... and of the time a fall onto it would take
const float SMALLEST = 1.0 / 4096.0;    //shortest step (as share of the step of the bodies)
const float REJOIN = 1.15;        //a body counts as part of the center again this far beyond where it stopped

vec4  path_from[BODIES], path_to[BODIES];   //every body at the start and end of the step
vec4  around[BODIES];                       //every body now, relative to the center of the particle
float mass[BODIES], radius[BODIES], near[BODIES], low[BODIES], period[BODIES], share[BODIES], calm2[BODIES];
float inside[BODIES + 1];                   //mass of S and all bodies nearer than each one

//position of a body between the two ends of the step (cubic curve through both)
vec4 between(vec4 a, vec4 b, float s, float dt) {
    float s2 = s * s, s3 = s2 * s;
    vec2 position = (2.0 * s3 - 3.0 * s2 + 1.0) * a.xy + (s3 - 2.0 * s2 + s) * dt * a.zw
                  + (3.0 * s2 - 2.0 * s3) * b.xy + (s3 - s2) * dt * b.zw;
    vec2 velocity = (6.0 * s2 - 6.0 * s) / dt * (a.xy - b.xy) + (3.0 * s2 - 4.0 * s + 1.0) * a.zw
                  + (3.0 * s2 - 2.0 * s) * b.zw;
    return vec4(position, velocity);
}

//one body at the time tau into the step
vec4 body_at(int k, int inner, float tau, float dt) {
    if (tau <= 0.0) return path_from[k];
    if (tau >= dt) return path_to[k];
    if (k >= inner) return between(path_from[k], path_to[k], tau / dt, dt);
    vec4 state = path_from[k];
    kepler_flow(state.xy, state.zw, inside[k + 1], tau);
    return state;
}

//calculates the bodies in play at the time tau, relative to the center the particle moves around
void locate(int count, int inner, int depth, int play, float tau, float dt) {
    vec4 offset = vec4(0.0);
    for (int k = depth; k < count; ++k) {
        if ((play & (1 << k)) == 0) continue;
        vec4 state = body_at(k, inner, tau, dt);
        around[k] = state + offset;
        if (k < inner) offset += share[k] * state;
    }
}

//pull of the bodies at x minus their pull on the center
vec2 pull(int count, int depth, int play, vec2 x) {
    vec2 sum = vec2(0.0);
    for (int k = depth; k < count; ++k) {
        if ((play & (1 << k)) == 0) continue;
        vec2 body = around[k].xy;
        vec2 d = body - x;
        float d2 = max(dot(d, d), radius[k] * radius[k]);
        float b2 = dot(body, body);
        sum += mass[k] * (d * inversesqrt(d2) / d2 - body * inversesqrt(b2) / b2);
    }
    return sum;
}

//squared distance of the origin from the line between a and b
float passing2(vec2 a, vec2 b) {
    vec2 along = b - a;
    float t = clamp(-dot(a, along) / max(dot(along, along), 1e-30), 0.0, 1.0);
    vec2 nearest = a + t * along;
    return dot(nearest, nearest);
}

void main()
{
    int group = int(gl_WorkGroupID.x);
    int column = u_first_column + group;
    int row = int(gl_LocalInvocationID.x);
    int cell = group * __ROWS__ + row;
    if (u_resume != 0 && map_particles[cell].at.w != 0) return;

    int count = min(u_body_count, BODIES);
    Column c = column_of(column);

    for (int k = 0; k < count; ++k) {
        MapBody body = map_bodies[k];
        mass[k] = body.info.x;
        radius[k] = body.info.y;
        inside[k] = body.info.w;
        period[k] = body.more.x;
        share[k] = body.more.y;
        near[k] = body.more.z;
        low[k] = body.more.w;
        //beyond this (squared) distance a fall onto the body takes longer than a whole step
        calm2[k] = pow(c.dt * c.dt * mass[k] / (FALL * FALL), 2.0 / 3.0);
    }
    inside[count] = map_bodies[count].info.w;

    //start position relative to the center the particle moves around
    float speed = u_speed_max * (float(row) + 0.5) / float(__ROWS__);
    vec4 centre = map_bodies[c.inner].centre;
    vec2 x = u_direction * c.r0 - centre.xy;
    vec2 v = u_sense * vec2(-u_direction.y, u_direction.x) * (speed / sqrt(c.r0)) - centre.zw;
    float nearest = length(x);
    float farthest = nearest;

    //follows as many orbits of its own as the column does of the start circle (if that is shorter)
    float size = 1.0 / max(2.0 - nearest * dot(v, v) / c.mu, 1e-6);   //semi-major axis divided by the start distance
    int steps = clamp(int(ceil(float(c.steps) * min(1.0, size * sqrt(size)))), 1, c.steps);
    int budget = steps + u_extra_steps;

    int depth = c.inner;
    int step = -1;
    float tau = c.dt;
    int turns = 0;
    if (u_resume != 0) {
        MapParticle particle = map_particles[cell];
        x = particle.motion.xy;
        v = particle.motion.zw;
        nearest = particle.seen.x;
        farthest = particle.seen.y;
        tau = particle.seen.z;
        step = particle.at.x;
        depth = particle.at.y;
        turns = particle.at.z;
    }
    float mu = inside[depth];
    for (int k = 0; k < count; ++k) {
        if (step >= 0) path_from[k] = map_paths[path_index(group, step, k)];
        path_to[k] = map_paths[path_index(group, step + 1, k)];
    }

    float outcome = 0.0;
    float when = 1.0;
    float hit = -1.0;
    bool done = false;
    bool located = false;
    int play = 0;
    vec2 acc = vec2(0.0);
    vec2 before[BODIES];

    for (int turn = 0; turn < u_turns; ++turn) {
        if (tau >= c.dt) {
            step += 1;
            if (step >= steps) { done = true; break; }
            tau = 0.0;
            for (int k = 0; k < count; ++k) {
                path_from[k] = path_to[k];
                path_to[k] = map_paths[path_index(group, step + 1, k)];
            }
        }
        if (turns >= budget) {
            hit = -2.0;
            when = (float(step) + tau / c.dt) / float(steps);
            done = true;
            break;
        }
        turns += 1;

        //bodies in play: all except those the particle is far inside the orbit of
        float r_start = length(x);
        int now_play = 0;
        for (int k = depth; k < count; ++k)
            if (k >= c.inner || r_start >= low[k]) now_play |= 1 << k;
        if (!located || now_play != play) {
            play = now_play;
            locate(count, c.inner, depth, play, tau, c.dt);
            acc = pull(count, depth, play, x);
            located = true;
        }

        //how long this step can be (shorter near a body)
        float h = c.dt - tau;
        float whole2 = h * h;
        for (int k = depth; k < count; ++k) {
            if ((play & (1 << k)) == 0) continue;
            vec2 d = around[k].xy - x;
            before[k] = d;
            float gap2 = dot(d, d);
            vec2 past = around[k].zw - v;
            float closing2 = dot(past, past);
            if (k < c.inner) h = min(h, period[k] / RESOLVE);
            if (gap2 * (ENCOUNTER * ENCOUNTER) < whole2 * closing2 || gap2 < calm2[k]) {
                float gap = sqrt(gap2);
                h = min(h, min(ENCOUNTER * max(gap, radius[k]) * inversesqrt(max(closing2, 1e-30)),
                               FALL * sqrt(gap2 * gap / mass[k])));
            }
        }
        h = max(h, c.dt * SMALLEST);
        float next = tau + h;
        if (next > c.dt * (1.0 - 0.5 * SMALLEST)) { next = c.dt; h = c.dt - tau; }

        v += acc * (0.5 * h);
        vec2 x_start = x, v_start = v;
        bool falling = dot(x, v) < 0.0;

        kepler_flow(x, v, mu, h);
        tau = next;
        locate(count, c.inner, depth, play, tau, c.dt);
        acc = pull(count, depth, play, x);

        //remembers the nearest and farthest point of the orbit if the step passed them
        float r = length(x);
        bool rising = dot(x, v) >= 0.0;
        float closest = r, widest = r;
        if (falling == rising || h * h * mu >= 1.2 * r_start * r_start * r_start) {
            float spin = x_start.x * v_start.y - x_start.y * v_start.x;
            float energy = 0.5 * dot(v_start, v_start) - mu / r_start;
            float e = sqrt(max(0.0, 1.0 + 2.0 * energy * spin * spin / (mu * mu)));
            bool closed = energy < 0.0 && e < 1.0;
            float turn_time = closed ? 6.2831853 * mu / (-2.0 * energy * sqrt(-2.0 * energy)) : 3.0e38;
            bool lap = h >= turn_time || (falling != rising && h > 0.5 * turn_time);
            if (lap || (falling && rising)) closest = min(r, spin * spin / mu / (1.0 + e));
            if (closed && (lap || (!falling && !rising))) widest = max(r, spin * spin / mu / (1.0 - e));
        }
        v += acc * (0.5 * h);
        nearest = min(nearest, closest);
        farthest = max(farthest, widest);

        if (depth == 0 && closest < u_impact) {
            outcome = 1.0;
        } else {
            for (int k = depth; k < count; ++k) {
                if ((play & (1 << k)) == 0) continue;
                vec2 after = around[k].xy - x;
                float reach2 = radius[k] * radius[k];
                if (min(dot(before[k], before[k]), dot(after, after)) < 16.0 * reach2
                        && passing2(before[k], after) < reach2) {
                    outcome = 3.0;
                    hit = float(k);
                    break;
                }
            }
            if (outcome == 0.0 && r > u_escape) outcome = 2.0;
            if (outcome == 0.0 && r > 2.0 && dot(x, v) > 0.0) {
                float enclosed = mu;
                for (int k = depth; k < count; ++k)
                    if ((play & (1 << k)) != 0 && dot(around[k].xy, around[k].xy) < r * r) enclosed += mass[k];
                if (dot(v, v) > 2.0 * enclosed / r) outcome = 2.0;
            }
        }
        if (outcome != 0.0) {
            when = (float(step) + tau / c.dt) / float(steps);
            done = true;
            break;
        }

        //closer than a body that counted as part of the center: it counts on its own again
        while (depth > 0 && r < near[depth - 1]) {
            depth -= 1;
            vec4 state = body_at(depth, c.inner, tau, c.dt);
            x += share[depth] * state.xy;
            v += share[depth] * state.zw;
            mu = inside[depth];
            r = length(x);
            located = false;
        }
        //far outside of it again: back into the center
        while (depth < c.inner && r > REJOIN * near[depth]) {
            vec4 state = body_at(depth, c.inner, tau, c.dt);
            x -= share[depth] * state.xy;
            v -= share[depth] * state.zw;
            depth += 1;
            mu = inside[depth];
            r = length(x);
            located = false;
        }
    }

    if (done) {
        if (outcome == 0.0) {
            //at the end: checks if it is leaving for good or if its orbit goes beyond the limit
            float r = length(x);
            play = 0;
            for (int k = depth; k < count; ++k)
                if (k >= c.inner || r >= low[k]) play |= 1 << k;
            locate(count, c.inner, depth, play, min(tau, c.dt), c.dt);
            float enclosed = mu;
            for (int k = depth; k < count; ++k)
                if ((play & (1 << k)) != 0 && dot(around[k].xy, around[k].xy) < r * r) enclosed += mass[k];
            float energy = 0.5 * dot(v, v) - enclosed / r;
            float spin = x.x * v.y - x.y * v.x;
            float e = sqrt(max(0.0, 1.0 + 2.0 * energy * spin * spin / (enclosed * enclosed)));
            bool leaving = energy >= 0.0 ? dot(x, v) > 0.0 : -0.5 * enclosed / energy * (1.0 + e) > u_escape;
            if (leaving) outcome = 2.0;
        }
        map_cells[column * __ROWS__ + row] = vec4(outcome, when, (farthest - nearest) / (farthest + nearest), hit);
    } else {
        atomicAdd(map_left[0], 1u);
    }
    map_particles[cell] = MapParticle(vec4(x, v), vec4(nearest, farthest, tau, 0.0), ivec4(step, depth, turns, int(done)));
}
""").replace("__ROWS__", str(MAP_ROWS)).replace("__BODIES__", str(MAP_BODIES)).replace("__STEPS__", str(MAP_STEPS))


#preview of the stability map in the view: the circle of the start distance around S, the start point and its direction
#drawn on the gpu from the position of S, so it needs no read-back and no upload
PREVIEW_VERTEX_SHADER = VERTEX_START + """
uniform int   u_body;        //S
uniform float u_distance;    //distance of the start from S (m)
uniform float u_length;      //length of the line that shows the start direction (px)

flat out vec2 v_center;      //S on screen (px)
flat out float v_radius;     //radius of the circle (px)

void main() {
    gl_Position = CULLED;
    v_center = vec2(0.0);
    v_radius = 0.0;
    if (!valid_body(u_body)) return;

    v_center = body_to_screen(u_body, camera_center());
    v_radius = u_distance * u_zoom;

    //bounds of everything, clipped to the screen
    float reach = v_radius + u_length + 8.0;
    vec2 lo = max(v_center - reach, vec2(-2.0));
    vec2 hi = min(v_center + reach, u_resolution + 2.0);
    if (lo.x >= hi.x || lo.y >= hi.y) return;

    gl_Position = screen_to_clip(mix(lo, hi, quad_corner() * 0.5 + 0.5));
}
"""


PREVIEW_FRAGMENT_SHADER = FRAGMENT_START + """
uniform vec2  u_direction;   //direction of the start (seen from S)
uniform vec2  u_tangent;     //direction it starts moving in
uniform float u_length;
uniform vec3  u_color;       //color of the start (shows what happens to it)

flat in vec2 v_center;
flat in float v_radius;

out vec4 fragColor;

const vec3 CIRCLE_COLOR = vec3(0.43, 0.51, 0.63);

void main() {
    vec2 frag = vec2(gl_FragCoord.x, u_resolution.y - gl_FragCoord.y);
    vec2 d = frag - v_center;

    //the circle in sixty dashes (not for circles that are too small or too big)
    float dash = fract(atan(d.y, d.x) / 6.2831853 * 60.0) < 0.5 ? 1.0 : 0.0;
    bool shown = v_radius > 6.0 && v_radius < 4.0 * max(u_resolution.x, u_resolution.y);
    float circle = shown ? clamp(1.0 - abs(length(d) - v_radius), 0.0, 1.0) * dash : 0.0;

    //the start: a small ring and a line in its direction
    vec2 p = frag - (v_center + u_direction * v_radius);
    vec2 along = u_tangent * u_length;
    float t = clamp(dot(p, along) / max(dot(along, along), 1e-6), 0.0, 1.0);
    float ring = clamp(1.5 - abs(length(p) - 4.0), 0.0, 1.0);
    float stroke = clamp(1.5 - length(p - along * t), 0.0, 1.0);
    float mark = max(ring, stroke);

    float alpha = max(circle, mark);
    if (alpha <= 0.003) discard;
    fragColor = vec4(mark >= circle ? u_color : CIRCLE_COLOR, alpha);
}
"""


### UI OVERLAY ###
#draws the pygame ui surface over the finished frame

OVERLAY_VERTEX_SHADER = """
#version 330 core

in vec2 in_quad_vertex;

out vec2 v_uv;

void main() {
    //pygame draws from the top, opengl from the bottom
    v_uv = vec2(in_quad_vertex.x * 0.5 + 0.5, 0.5 - in_quad_vertex.y * 0.5);
    gl_Position = vec4(in_quad_vertex, 0.0, 1.0);
}
"""


OVERLAY_FRAGMENT_SHADER = """
#version 330 core

in vec2 v_uv;

uniform sampler2D u_overlay;

out vec4 fragColor;

void main() {
    vec4 texel = texture(u_overlay, v_uv);
    if (texel.a <= 0.002) discard;
    fragColor = texel;
}
"""
