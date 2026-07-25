#include "ufbx_bridge.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ufbx.h"

static void set_error(char *err, int err_cap, const char *message)
{
    if (!err || err_cap <= 0) {
        return;
    }
    snprintf(err, (size_t)err_cap, "%s", message ? message : "unknown error");
}

static char *dup_cstr(const char *src)
{
    size_t len;
    char *dst;
    if (!src) {
        src = "";
    }
    len = strlen(src);
    dst = (char *)malloc(len + 1);
    if (!dst) {
        return NULL;
    }
    memcpy(dst, src, len + 1);
    return dst;
}

static void append_vec3(float *dst, int *cursor, ufbx_vec3 v)
{
    int i = *cursor;
    dst[i + 0] = (float)v.x;
    dst[i + 1] = (float)v.y;
    dst[i + 2] = (float)v.z;
    *cursor = i + 3;
}

static void append_vec2(float *dst, int *cursor, ufbx_vec2 v)
{
    int i = *cursor;
    dst[i + 0] = (float)v.x;
    dst[i + 1] = (float)v.y;
    *cursor = i + 2;
}

static ufbx_vec3 normalize_vec3(ufbx_vec3 n)
{
    double len = sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    if (len <= 1e-12) {
        ufbx_vec3 up = { 0.0, 1.0, 0.0 };
        return up;
    }
    n.x /= len;
    n.y /= len;
    n.z /= len;
    return n;
}

static ufbx_vec3 face_normal(ufbx_vec3 a, ufbx_vec3 b, ufbx_vec3 c)
{
    ufbx_vec3 ab = { b.x - a.x, b.y - a.y, b.z - a.z };
    ufbx_vec3 ac = { c.x - a.x, c.y - a.y, c.z - a.z };
    ufbx_vec3 n = {
        ab.y * ac.z - ab.z * ac.y,
        ab.z * ac.x - ab.x * ac.z,
        ab.x * ac.y - ab.y * ac.x,
    };
    return normalize_vec3(n);
}

static ufbx_mesh *pick_largest_mesh(ufbx_scene *scene)
{
    ufbx_mesh *best = NULL;
    size_t best_tris = 0;
    size_t i;
    for (i = 0; i < scene->meshes.count; i++) {
        ufbx_mesh *mesh = scene->meshes.data[i];
        if (!mesh) {
            continue;
        }
        if (mesh->num_triangles > best_tris) {
            best = mesh;
            best_tris = mesh->num_triangles;
        }
    }
    return best;
}

/* World matrix of the first instance, translation stripped so the app places it. */
static void mesh_bake_matrix(ufbx_mesh *mesh, ufbx_matrix *out_pos, ufbx_matrix *out_nrm)
{
    ufbx_matrix world = ufbx_identity_matrix;
    if (mesh->instances.count > 0 && mesh->instances.data[0] != NULL) {
        world = mesh->instances.data[0]->geometry_to_world;
    }
    world.cols[3].x = 0.0;
    world.cols[3].y = 0.0;
    world.cols[3].z = 0.0;
    *out_pos = world;
    *out_nrm = ufbx_matrix_for_normals(&world);
}

static int triangulate_mesh(ufbx_mesh *mesh, BridgeMesh *out_mesh, char *err, int err_cap)
{
    size_t max_tris = mesh->num_triangles;
    size_t max_corners = max_tris * 3;
    size_t tri_cap = mesh->max_face_triangles * 3;
    uint32_t *tri_indices = NULL;
    float *positions = NULL;
    float *normals = NULL;
    float *texcoords = NULL;
    int pos_i = 0;
    int nrm_i = 0;
    int uv_i = 0;
    size_t fi;
    int have_normals = mesh->vertex_normal.exists;
    int have_uvs = mesh->vertex_uv.exists;
    ufbx_matrix pos_mat;
    ufbx_matrix nrm_mat;

    if (max_corners == 0) {
        set_error(err, err_cap, "mesh has no triangles");
        return 1;
    }

    mesh_bake_matrix(mesh, &pos_mat, &nrm_mat);

    tri_indices = (uint32_t *)malloc(sizeof(uint32_t) * (tri_cap > 0 ? tri_cap : 3));
    positions = (float *)malloc(sizeof(float) * max_corners * 3);
    normals = (float *)malloc(sizeof(float) * max_corners * 3);
    texcoords = (float *)malloc(sizeof(float) * max_corners * 2);
    if (!tri_indices || !positions || !normals || !texcoords) {
        free(tri_indices);
        free(positions);
        free(normals);
        free(texcoords);
        set_error(err, err_cap, "out of memory while triangulating mesh");
        return 1;
    }

    for (fi = 0; fi < mesh->faces.count; fi++) {
        ufbx_face face = mesh->faces.data[fi];
        uint32_t num_tris;
        uint32_t ti;
        if (face.num_indices < 3) {
            continue;
        }
        num_tris = ufbx_triangulate_face(
            tri_indices,
            tri_cap > 0 ? tri_cap : 3,
            mesh,
            face
        );
        for (ti = 0; ti < num_tris; ti++) {
            uint32_t c0 = tri_indices[ti * 3 + 0];
            uint32_t c1 = tri_indices[ti * 3 + 1];
            uint32_t c2 = tri_indices[ti * 3 + 2];
            ufbx_vec3 p0 = ufbx_transform_position(
                &pos_mat, ufbx_get_vertex_vec3(&mesh->vertex_position, c0)
            );
            ufbx_vec3 p1 = ufbx_transform_position(
                &pos_mat, ufbx_get_vertex_vec3(&mesh->vertex_position, c1)
            );
            ufbx_vec3 p2 = ufbx_transform_position(
                &pos_mat, ufbx_get_vertex_vec3(&mesh->vertex_position, c2)
            );
            ufbx_vec3 n0;
            ufbx_vec3 n1;
            ufbx_vec3 n2;
            ufbx_vec2 uv0 = { 0.0, 0.0 };
            ufbx_vec2 uv1 = { 0.0, 0.0 };
            ufbx_vec2 uv2 = { 0.0, 0.0 };

            append_vec3(positions, &pos_i, p0);
            append_vec3(positions, &pos_i, p1);
            append_vec3(positions, &pos_i, p2);

            if (have_normals) {
                n0 = normalize_vec3(ufbx_transform_direction(
                    &nrm_mat, ufbx_get_vertex_vec3(&mesh->vertex_normal, c0)
                ));
                n1 = normalize_vec3(ufbx_transform_direction(
                    &nrm_mat, ufbx_get_vertex_vec3(&mesh->vertex_normal, c1)
                ));
                n2 = normalize_vec3(ufbx_transform_direction(
                    &nrm_mat, ufbx_get_vertex_vec3(&mesh->vertex_normal, c2)
                ));
            } else {
                n0 = n1 = n2 = face_normal(p0, p1, p2);
            }
            append_vec3(normals, &nrm_i, n0);
            append_vec3(normals, &nrm_i, n1);
            append_vec3(normals, &nrm_i, n2);

            if (have_uvs) {
                uv0 = ufbx_get_vertex_vec2(&mesh->vertex_uv, c0);
                uv1 = ufbx_get_vertex_vec2(&mesh->vertex_uv, c1);
                uv2 = ufbx_get_vertex_vec2(&mesh->vertex_uv, c2);
            }
            append_vec2(texcoords, &uv_i, uv0);
            append_vec2(texcoords, &uv_i, uv1);
            append_vec2(texcoords, &uv_i, uv2);
        }
    }

    free(tri_indices);

    if (pos_i < 9) {
        free(positions);
        free(normals);
        free(texcoords);
        set_error(err, err_cap, "mesh produced no triangles");
        return 1;
    }

    out_mesh->name = dup_cstr(mesh->name.data ? mesh->name.data : "mesh");
    if (!out_mesh->name) {
        free(positions);
        free(normals);
        free(texcoords);
        set_error(err, err_cap, "out of memory copying mesh name");
        return 1;
    }
    out_mesh->positions = positions;
    out_mesh->normals = normals;
    out_mesh->texcoords = texcoords;
    out_mesh->vertex_count = pos_i / 3;
    out_mesh->triangle_count = out_mesh->vertex_count / 3;
    return 0;
}

UFBX_BRIDGE_API int bridge_load_fbx(
    const char *path,
    BridgeMesh *out_mesh,
    char *err,
    int err_cap
)
{
    ufbx_load_opts opts;
    ufbx_error error;
    ufbx_scene *scene;
    ufbx_mesh *mesh;
    int rc;

    if (!path || !out_mesh) {
        set_error(err, err_cap, "invalid arguments");
        return 1;
    }

    memset(out_mesh, 0, sizeof(*out_mesh));
    memset(&opts, 0, sizeof(opts));
    /* Metres, Y-up. ADJUST_TRANSFORMS puts unit/axis conversion into node
       matrices (and undoes Blender's common *100 export scale). We then bake
       the first instance's geometry_to_world into the triangle soup. */
    opts.target_axes = ufbx_axes_right_handed_y_up;
    opts.target_unit_meters = 1.0f;
    opts.space_conversion = UFBX_SPACE_CONVERSION_ADJUST_TRANSFORMS;
    opts.geometry_transform_handling = UFBX_GEOMETRY_TRANSFORM_HANDLING_MODIFY_GEOMETRY;

    scene = ufbx_load_file(path, &opts, &error);
    if (!scene) {
        set_error(
            err,
            err_cap,
            error.description.data ? error.description.data : "ufbx_load_file failed"
        );
        return 1;
    }

    mesh = pick_largest_mesh(scene);
    if (!mesh) {
        ufbx_free_scene(scene);
        set_error(err, err_cap, "fbx contains no meshes");
        return 1;
    }

    rc = triangulate_mesh(mesh, out_mesh, err, err_cap);
    ufbx_free_scene(scene);
    return rc;
}

UFBX_BRIDGE_API void bridge_free_mesh(BridgeMesh *mesh)
{
    if (!mesh) {
        return;
    }
    free(mesh->name);
    free(mesh->positions);
    free(mesh->normals);
    free(mesh->texcoords);
    memset(mesh, 0, sizeof(*mesh));
}
