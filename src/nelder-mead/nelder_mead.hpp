
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define RHO 1.0
#define CHI 2.0
#define GAMMA 0.5
#define SIGMA 0.5
#define SQUARE(x) ((x) * (x))

// define a generic point containing a position (x) and a value (fx)
template <typename REAL>
struct point_t {
  REAL *x;
  REAL fx;
};

// define a simplex struct containing an array of n+1 points (p)
// each having dimension (n)
template <typename REAL>
struct simplex_t {
  point_t<REAL> *p;
  int n;
};

// define optimization settings
template <typename REAL>
struct optimset_t {
  REAL tolx;
  REAL tolf;
  int max_iter;
  int max_eval;
  int verbose;
};


// typedef void (*fun_t)(int, point_t *, const void *);
template <typename REAL>
using fun_t = void (*)(int, point_t<REAL> *, const void *);



//-----------------------------------------------------------------------------
// Simplex sorting
//-----------------------------------------------------------------------------

template <typename REAL>
int compare(const void *arg1, const void *arg2) {
  const REAL fx1 = (((point_t<REAL> *)arg1)->fx);
  const REAL fx2 = (((point_t<REAL> *)arg2)->fx);

  if (fx1 == fx2) {
    return 0;
  } else {
    return (fx1 < fx2) ? -1 : 1;
  }
}

template <typename REAL>
void simplex_sort(simplex_t<REAL> *simplex) {
  qsort((void *)(simplex->p), simplex->n + 1, sizeof(point_t<REAL>), compare<REAL>);
}

//-----------------------------------------------------------------------------
// Get centroid (average position) of simplex
//-----------------------------------------------------------------------------

template <typename REAL>
void get_centroid(const simplex_t<REAL> *simplex, point_t<REAL> *centroid) {
  for (int j = 0; j < simplex->n; j++) {
    centroid->x[j] = 0;
    for (int i = 0; i < simplex->n; i++) {
      centroid->x[j] += simplex->p[i].x[j];
    }
    centroid->x[j] /= simplex->n;
  }
}

//-----------------------------------------------------------------------------
// Asses if simplex satisfies the minimization requirements
//-----------------------------------------------------------------------------

template <typename REAL>
int continue_minimization(const simplex_t<REAL> *simplex, int eval_count,
                          int iter_count, const optimset_t<REAL> *optimset) {
  if (eval_count > optimset->max_eval || iter_count > optimset->max_iter) {
    // stop if #evals or #iters are greater than the max allowed
    return 0;
  }
  REAL condx = -1.0;
  REAL condf = -1.0;
  for (int i = 1; i < simplex->n + 1; i++) {
    const REAL temp = fabs(simplex->p[0].fx - simplex->p[i].fx);
    if (condf < temp) {
      condf = temp;
    }
  }
  for (int i = 1; i < simplex->n + 1; i++) {
    for (int j = 0; j < simplex->n; j++) {
      const REAL temp = fabs(simplex->p[0].x[j] - simplex->p[i].x[j]);
      if (condx < temp) {
        condx = temp;
      }
    }
  }
  // continue if both tolx or tolf condition is not met
  return condx > optimset->tolx || condf > optimset->tolf;
}

//-----------------------------------------------------------------------------
// Update current point
//-----------------------------------------------------------------------------

template <typename REAL>
void update_point(const simplex_t<REAL> *simplex, const point_t<REAL> *centroid,
                  REAL lambda, point_t<REAL> *point) {
  const int n = simplex->n;
  for (int j = 0; j < n; j++) {
    point->x[j] = (1.0 + lambda) * centroid->x[j] - lambda * simplex->p[n].x[j];
  }
}

//-----------------------------------------------------------------------------
// Simple point_t manipulation utlities
//-----------------------------------------------------------------------------

template <typename REAL>
void copy_point(int n, const point_t<REAL> *src, point_t<REAL> *dst) {
  for (int j = 0; j < n; j++) {
    dst->x[j] = src->x[j];
  }
  dst->fx = src->fx;
}

template <typename REAL>
void swap_points(int n, point_t<REAL> *p1, point_t<REAL> *p2) {
  REAL temp;
  for (int j = 0; j < n; j++) {
    temp = p1->x[j];
    p1->x[j] = p2->x[j];
    p2->x[j] = temp;
  }
  temp = p1->fx;
  p1->fx = p2->fx;
  p2->fx = temp;
}




//-----------------------------------------------------------------------------
// Main function
// - n is the dimension of the data
// - start is the initial point (unchanged in output)
// - solution is the minimizer
// - cost_function is a pointer to a fun_t type function
// - args are the optional arguments of cost_function
// - optimset are the optimisation settings
//-----------------------------------------------------------------------------

template <typename REAL>
void nelder_mead(int n, const point_t<REAL> *start, point_t<REAL> *solution,
                 fun_t<REAL> cost_function, const void *args,
                 const optimset_t<REAL> *optimset) {
  // internal points
  point_t<REAL> point_r;
  point_t<REAL> point_e;
  point_t<REAL> point_c;
  point_t<REAL> centroid;

  // allocate memory for internal points
  point_r.x = (REAL*) malloc(n * sizeof(REAL));
  point_e.x = (REAL*) malloc(n * sizeof(REAL));
  point_c.x = (REAL*) malloc(n * sizeof(REAL));
  centroid.x = (REAL*) malloc(n * sizeof(REAL));

  int iter_count = 0;
  int eval_count = 0;

  // initial simplex has size n + 1 where n is the dimensionality pf the data
  simplex_t<REAL> simplex;
  simplex.n = n;
  simplex.p = (point_t<REAL>*) malloc((n + 1) * sizeof(point_t<REAL>));
  for (int i = 0; i < n + 1; i++) {
    simplex.p[i].x = (REAL*) malloc(n * sizeof(REAL));
    for (int j = 0; j < n; j++) {
      simplex.p[i].x[j] =
          (i - 1 == j) ? (start->x[j] != 0.0 ? 1.05 * start->x[j] : 0.00025)
                       : start->x[j];
    }
    cost_function(n, simplex.p + i, args);
    eval_count++;
  }
  // sort points in the simplex so that simplex.p[0] is the point having
  // minimum fx and simplex.p[n] is the one having the maximum fx
  simplex_sort(&simplex);
  // ompute the simplex centroid
  get_centroid(&simplex, &centroid);
  iter_count++;

  // continue minimization until stop conditions are met
  while (continue_minimization(&simplex, eval_count, iter_count, optimset)) {
    int shrink = 0;

    if (optimset->verbose) {
      printf("Iteration %04d     ", iter_count);
    }
    update_point(&simplex, &centroid, RHO, &point_r);
    cost_function(n, &point_r, args);
    eval_count++;
    if (point_r.fx < simplex.p[0].fx) {
      update_point(&simplex, &centroid, RHO * CHI, &point_e);
      cost_function(n, &point_e, args);
      eval_count++;
      if (point_e.fx < point_r.fx) {
        // expand
        if (optimset->verbose) {
          printf("expand          ");
        }
        copy_point(n, &point_e, simplex.p + n);
      } else {
        // reflect
        if (optimset->verbose) {
          printf("reflect         ");
        }
        copy_point(n, &point_r, simplex.p + n);
      }
    } else {
      if (point_r.fx < simplex.p[n - 1].fx) {
        // reflect
        if (optimset->verbose) {
          printf("reflect         ");
        }
        copy_point(n, &point_r, simplex.p + n);
      } else {
        if (point_r.fx < simplex.p[n].fx) {
          update_point(&simplex, &centroid, RHO * GAMMA, &point_c);
          cost_function(n, &point_c, args);
          eval_count++;
          if (point_c.fx <= point_r.fx) {
            // contract outside
            if (optimset->verbose) {
              printf("contract out    ");
            }
            copy_point(n, &point_c, simplex.p + n);
          } else {
            // shrink
            if (optimset->verbose) {
              printf("shrink         ");
            }
            shrink = 1;
          }
        } else {
          update_point(&simplex, &centroid, -GAMMA, &point_c);
          cost_function(n, &point_c, args);
          eval_count++;
          if (point_c.fx <= simplex.p[n].fx) {
            // contract inside
            if (optimset->verbose) {
              printf("contract in     ");
            }
            copy_point(n, &point_c, simplex.p + n);
          } else {
            // shrink
            if (optimset->verbose) {
              printf("shrink         ");
            }
            shrink = 1;
          }
        }
      }
    }
    if (shrink) {
      for (int i = 1; i < n + 1; i++) {
        for (int j = 0; j < n; j++) {
          simplex.p[i].x[j] = simplex.p[0].x[j] +
                              SIGMA * (simplex.p[i].x[j] - simplex.p[0].x[j]);
        }
        cost_function(n, simplex.p + i, args);
        eval_count++;
      }
      simplex_sort(&simplex);
    } else {
      for (int i = n - 1; i >= 0 && simplex.p[i + 1].fx < simplex.p[i].fx; i--) {
        swap_points(n, simplex.p + (i + 1), simplex.p + i);
      }
    }
    get_centroid(&simplex, &centroid);
    iter_count++;
    if (optimset->verbose) {
      // print current minimum
      printf("[ ");
      for (int i = 0; i < n; i++) {
        printf("%.2f ", simplex.p[0].x[i]);
      }
      printf("]    %.2f \n", simplex.p[0].fx);
    }
  }
  
  // save solution in output argument
  solution->x = (REAL*) malloc(n * sizeof(REAL));
  copy_point(n, simplex.p + 0, solution);

  // free memory
  free(centroid.x);
  free(point_r.x);
  free(point_e.x);
  free(point_c.x);
  for (int i = 0; i < n + 1; i++) {
    free(simplex.p[i].x);
  }
  free(simplex.p);
}
