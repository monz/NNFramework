#include "mex.h"
#include "matrix.h"

/*
 *  This is a fast and simple implementation of http://de.mathworks.com/help/matlab/ref/num2cell.html.
 *  But it only supports column based conversion (see MATLAB help of num2cell with dimension = 1).
 *
 *  For Example a 9x5 matrix gets converted to a 5x1 cell array.
 *  Each cell contains a column vector of size 9x1.
 *
 *  Usage: myNum2Cell(values)
 */

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mxArray *cell_array_ptr, *mycell;
    mwIndex i, j;
    size_t nrows, ncols;

    /* Check for proper number of input and output arguments */
    if (nrhs < 1) {
        mexErrMsgIdAndTxt("MyNum2Cell:myNum2Cell:minrhs",
                "At least one input argument required.");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("MyNum2Cell:myNum2Cell:maxlhs",
                "Exactly one output argument required.");
    }

    /* make sure the input argument is type double */
    if( !mxIsDouble(prhs[0]) ||
         mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MyNum2Cell:myNum2Cell:notDouble",
                "Input matrix must be type double.");
    }

    /* get number of rows */
    nrows = mxGetM(prhs[0]);
    /* get number of cols */
    ncols = mxGetN(prhs[0]);

    /* Create a N x 1 cell mxArray. */
    cell_array_ptr = mxCreateCellMatrix(ncols,1);

    /* Fill cells with matrix values */
    mwSize ndim = 2;
    const mwSize dims[2] = {nrows, 1};
    double *mycell_data;
    double *input_matrix = mxGetPr(prhs[0]);
    for (j = 0; j < ncols; j++) {
        mycell = mxCreateNumericArray(ndim, dims, mxDOUBLE_CLASS, mxREAL);
        mycell_data = mxGetPr(mycell);
        for (i = 0; i < nrows; i++) {
            mycell_data[i] = input_matrix[i+(j*nrows)];
        }
        mxSetCell(cell_array_ptr, j, mxDuplicateArray(mycell));
        mxDestroyArray(mycell);
    }
    plhs[0] = cell_array_ptr;
}