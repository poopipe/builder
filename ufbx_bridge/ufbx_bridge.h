#ifndef BUILDER_UFBX_BRIDGE_H
#define BUILDER_UFBX_BRIDGE_H

#ifdef _WIN32
#ifdef UFBX_BRIDGE_EXPORTS
#define UFBX_BRIDGE_API __declspec(dllexport)
#else
#define UFBX_BRIDGE_API __declspec(dllimport)
#endif
#else
#define UFBX_BRIDGE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BridgeMesh {
    char *name;
    float *positions;
    float *normals;
    float *texcoords;
    int vertex_count;
    int triangle_count;
} BridgeMesh;

/* returns 0 on success, non-zero on failure; err is optional UTF-8 message buffer */
UFBX_BRIDGE_API int bridge_load_fbx(
    const char *path,
    BridgeMesh *out_mesh,
    char *err,
    int err_cap
);

UFBX_BRIDGE_API void bridge_free_mesh(BridgeMesh *mesh);

#ifdef __cplusplus
}
#endif

#endif
