/********************************************************************
 ********************************************************************
 **
 ** libhungarian by Cyrill Stachniss, 2004
 **
 **
 ** Solving the Minimum Assignment Problem using the
 ** Hungarian Method.
 **
 ** ** This file may be freely copied and distributed! **
 **
 ** Parts of the used code was originally provided by the
 ** "Stanford GraphGase", but I made changes to this code.
 ** As asked by  the copyright node of the "Stanford GraphGase",
 ** I hereby proclaim that this file are *NOT* part of the
 ** "Stanford GraphGase" distrubition!
 **
 ** This file is distributed in the hope that it will be useful,
 ** but WITHOUT ANY WARRANTY; without even the implied
 ** warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 ** PURPOSE.
 **
 ********************************************************************
 ********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "hungarian.h"

int main() {

    hungarian_problem_t p;

    /* an example cost matrix */
    /*int r[4*3] =  {  100, 1, 1,
    100, 2, 2,
    1, 0, 0,
    0, 2, 0 };
    int** m = array_to_matrix(r,4,3);*/

    /** Distance matrix **/


    int r[5*4] = {10,19,8,15,
                10,18,7,17,
                13,16,9,14,
                12,19,8,18,
                14,17,10,19};
    int** m= hungarian_array_to_matrix(r,5,4);

    /* initialize the hungarian_problem using the cost matrix*/
    //int matrix_size = hungarian_init(&p, m , 4,3, HUNGARIAN_MODE_MINIMIZE_COST) ;
    int matrix_size = hungarian_init(&p, m , 5,4, HUNGARIAN_MODE_MINIMIZE_COST) ;

    fprintf(stderr, "assignement matrix has now a size %d rows and %d columns.\n\n",  matrix_size,matrix_size);

    /* some output */
    fprintf(stderr, "cost-matrix:");
    hungarian_print_costmatrix(&p);

    /* solve the assignement problem */
    hungarian_solve(&p);

    /* some output */
    fprintf(stderr, "assignment:");
    hungarian_print_assignment(&p);

    /* free used memory */
    hungarian_free(&p);

    int idx;
    for (idx=0; idx < 4; idx+=1) {
        free(m[idx]);
    }
    free(m);

    return 0;
}
