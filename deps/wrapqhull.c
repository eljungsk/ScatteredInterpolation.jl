#include "libqhull_r/qhull_ra.h"

/*
 * This function computes the Delaunay triangulation of a set of points.
 *
 * Output:
 *          nFacets:    Number of facets in the triangulation.
 *          vertexList: (nDims+1)*nFacets matrix of vertex indices. Each column corresponds to a facet.
 *          neigbourList: 
 *
 */
void delaunay(
                int **vertexList,
                int **neigbourList,
                int **facetList,
                int *nFacets,
                double *points,
                int nDims,
                int nPoints
                ) {

    // Code heavily influenced by qhull/user_eg2_r.c
    char options [2000];
    int curlong, totlong;
    qhT qh_qh;
    qhT *qh = &qh_qh;

    QHULL_LIB_CHECK

    qh_init_A(qh, NULL, stdout, stderr, 0, NULL);
    int exitcode = setjmp(qh->errexit);

    // This is apparently a good set of flags
    if (nDims <= 3) {
        sprintf(options, "qhull d Qt Qbb Qc Qz");
    } else {
        sprintf(options, "qhull d Qt Qbb Qc Qx");
    }

    if (!exitcode) {

        qh->NOerrexit = False;
        qh_initflags(qh, options);
        qh->PROJECTdelaunay = True;
        qh_init_B(qh, points, nPoints, nDims, False);
        qh_qhull(qh);
        qh_triangulate(qh);
        qh_check_output(qh);
    }
    qh->NOerrexit = True;

    facetT *facet, *neighbor, **neighborp;
    vertexT *vertex, **vertexp;
    
    // Count the number of facets and allocate memory for the vertex list
    *nFacets = 0;
    FORALLfacets {
        if (!facet->upperdelaunay)
            (*nFacets)++;
    }
    // These pointers will later be returned to julia
    *facetList = malloc(*nFacets*sizeof(int));
    *vertexList = malloc(*nFacets*(nDims+1)*sizeof(int));
    *neigbourList = malloc(*nFacets*(nDims+1)*sizeof(int));
    
    // Get the indices for the vertices of each facet and the neighbours
    int i = 0, j = 0;
    FORALLfacets {

        if (!facet->upperdelaunay) {

            // Add the face id to the list of facets
            (*facetList)[i] = 1 + facet->id;

            // Get the vertices
            j = 0;
            FOREACHvertex_(facet->vertices) {

                (*vertexList)[(nDims+1)*i + j] = 1 + qh_pointid(qh, vertex->point);
                j++;
            }

            // Get the id:s of neighbour facets
            j = 0;
            FOREACHneighbor_(facet) {
                
                // Filter out the non-interesting halfspace
                if (!neighbor->upperdelaunay) {
                    (*neigbourList)[(nDims+1)*i + j] = 1 + neighbor->id;
                } else {
                    (*neigbourList)[(nDims+1)*i + j] = 0;
                }

                j++;
            }

            i++;
        }
    }

    // Clean up memory
    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);
    if (curlong || totlong) {
        fprintf(stderr, "qhull warning: did not free %d bytes of long memory (%d pieces)\n",
            totlong, curlong);
    }
    return;
}
