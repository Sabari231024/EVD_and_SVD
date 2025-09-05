#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <jpeglib.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>



gsl_matrix* imageread_grayscale(const char *file, int N) {
FILE *infile = fopen(file, "rb");
struct jpeg_decompress_struct cinfo;
struct jpeg_error_mgr jerr;
cinfo.err = jpeg_std_error(&jerr);
jpeg_create_decompress(&cinfo);
jpeg_stdio_src(&cinfo, infile);
jpeg_read_header(&cinfo, TRUE);
jpeg_start_decompress(&cinfo);
int row_stride = cinfo.output_width * cinfo.output_components;
JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);
gsl_matrix *mat = gsl_matrix_calloc(N, N);
int row = 0;
while (cinfo.output_scanline < cinfo.output_height && row < N) {
jpeg_read_scanlines(&cinfo, buffer, 1);
for (int col = 0; col < N && col < cinfo.output_width; col++) {
gsl_matrix_set(mat, row, col, (double)buffer[0][col]);
}
row++;
}
jpeg_finish_decompress(&cinfo);
jpeg_destroy_decompress(&cinfo);
fclose(infile);
return mat;
}

static void conj_transpose(const gsl_matrix_complex *B, gsl_matrix_complex *BH) { // thisfunction is used to get transpose , if complex do complex conjugate transpose
const size_t N = B->size1;
const size_t k = B->size2;
for (size_t i = 0; i < N; i++) {
for (size_t j = 0; j < k; j++) {
gsl_complex z = gsl_matrix_complex_get(B, i, j);
gsl_matrix_complex_set(BH, j, i, gsl_complex_conjugate(z));
}}}

double EVD_reconstruct_topk(const char *filename, int N, int k) {
gsl_matrix *A = imageread(filename, N);
gsl_matrix *A_copy = gsl_matrix_alloc(N, N);
gsl_matrix_memcpy(A_copy, A);
gsl_vector_complex *eval = gsl_vector_complex_alloc(N);
gsl_matrix_complex *evec = gsl_matrix_complex_alloc(N, N);
gsl_eigen_nonsymmv_workspace *w = gsl_eigen_nonsymmv_alloc(N);
gsl_eigen_nonsymmv_free(w);
gsl_eigen_nonsymmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_DESC);
gsl_matrix_complex *Bk = gsl_matrix_complex_alloc(N, k);
gsl_matrix_complex *Lk = gsl_matrix_complex_calloc(k, k); //this is an Diag matrix with eigen values
for (int j = 0; j < k; j++) {
gsl_complex lambda = gsl_vector_complex_get(eval, j);
gsl_matrix_complex_set(Lk, j, j, lambda);
for (int i = 0; i < N; i++) {
gsl_complex v = gsl_matrix_complex_get(evec, i, j);
gsl_matrix_complex_set(Bk, i, j, v);
}}
gsl_matrix_complex *BH = gsl_matrix_complex_alloc(k, N);
conj_transpose(Bk, BH);
gsl_matrix_complex *C = gsl_matrix_complex_calloc(k, k);
gsl_blas_zgemm(CblasNoTrans, CblasNoTrans,GSL_COMPLEX_ONE, BH, Bk,GSL_COMPLEX_ZERO, C);
double trace_real = 0.0;
for (int i = 0; i < k; i++) {
gsl_complex cii = gsl_matrix_complex_get(C, i, i);
trace_real += GSL_REAL(cii);
}
double alpha = (k > 0) ? (1e-8 * trace_real / (double)k) : 1e-8;
if (alpha <= 0.0) alpha = 1e-8;
for (int i = 0; i < k; i++) {
gsl_complex cii = gsl_matrix_complex_get(C, i, i);
cii = gsl_complex_add_real(cii, alpha);
gsl_matrix_complex_set(C, i, i, cii);
}
gsl_matrix_complex *C_copy = gsl_matrix_complex_alloc(k, k);
gsl_matrix_complex_memcpy(C_copy, C);
gsl_matrix_complex *Ak = gsl_matrix_complex_calloc(N, N);
int cholesky_status = gsl_linalg_complex_cholesky_decomp(C_copy);
if (cholesky_status != 0) {
gsl_permutation *p = gsl_permutation_alloc(k);
int signum;
gsl_matrix_complex *C_inv = gsl_matrix_complex_alloc(k, k);
gsl_linalg_complex_LU_invert(C, p, C_inv);
gsl_matrix_complex *Bpinv = gsl_matrix_complex_alloc(k, N);
gsl_blas_zgemm(CblasNoTrans, CblasNoTrans,GSL_COMPLEX_ONE, C_inv, BH,GSL_COMPLEX_ZERO, Bpinv);
gsl_matrix_complex *tmp = gsl_matrix_complex_alloc(k, N);
gsl_blas_zgemm(CblasNoTrans, CblasNoTrans,GSL_COMPLEX_ONE, Lk, Bpinv,GSL_COMPLEX_ZERO, tmp);
gsl_blas_zgemm(CblasNoTrans, CblasNoTrans,GSL_COMPLEX_ONE, Bk, tmp,GSL_COMPLEX_ZERO, Ak);
gsl_permutation_free(p);
gsl_matrix_complex_free(C_inv);
gsl_matrix_complex_free(Bpinv);
gsl_matrix_complex_free(tmp);
} 
else {
gsl_linalg_complex_cholesky_invert(C_copy); 
gsl_matrix_complex *Bpinv = gsl_matrix_complex_alloc(k, N);
gsl_blas_zgemm(CblasNoTrans, CblasNoTrans,GSL_COMPLEX_ONE, C_copy, BH,GSL_COMPLEX_ZERO, Bpinv);
gsl_matrix_complex *tmp = gsl_matrix_complex_alloc(k, N);
gsl_blas_zgemm(CblasNoTrans, CblasNoTrans,GSL_COMPLEX_ONE, Lk, Bpinv,GSL_COMPLEX_ZERO, tmp);
gsl_blas_zgemm(CblasNoTrans, CblasNoTrans,GSL_COMPLEX_ONE, Bk, tmp,GSL_COMPLEX_ZERO, Ak);
gsl_matrix_complex_free(Bpinv);
gsl_matrix_complex_free(tmp);
}
double err = 0.0;
for (int ii = 0; ii < N; ii++) {
for (int jj = 0; jj < N; jj++) {
double orig = gsl_matrix_get(A_copy, ii, jj);
gsl_complex z = gsl_matrix_complex_get(Ak, ii, jj);
double recon_real = GSL_REAL(z);
err += fabs(orig - recon_real);
        }}
gsl_matrix_complex_free(Ak);
gsl_matrix_complex_free(C_copy);
gsl_matrix_complex_free(C);
gsl_matrix_complex_free(BH);
gsl_matrix_complex_free(Lk);
gsl_matrix_complex_free(Bk);
gsl_matrix_complex_free(evec);
gsl_vector_complex_free(eval);
gsl_matrix_free(A_copy);
gsl_matrix_free(A);
return err;
}

int main(int argc, char *argv[]) {
    const char *filename = argv[1];
    int N = 256;       
    FILE *gp = popen("gnuplot", "w");
    if (!gp) {
        fprintf(stderr, "Error: could not open gnuplot.\n");
        return 1;
    }//i took these lines from internet
    fprintf(gp, "set terminal pngcairo size 800,600 enhanced font 'Arial,12'\n");
    fprintf(gp, "set output 'reconstruction_error.png'\n");
    fprintf(gp, "set title 'Reconstruction Error vs k'\n");
    fprintf(gp, "set xlabel 'k (Number of eigenvalues used)'\n");
    fprintf(gp, "set ylabel 'Reconstruction Error'\n");
    fprintf(gp, "set grid\n");
    fprintf(gp, "plot '-' with linespoints title 'Error'\n");
    for (int i = 1; i <= N; i=i+2) {
        double error = EVD_reconstruct_topk(filename, N, i);
        fprintf(gp, "%d %f\n", i, error);
        printf("k = %d, error = %.6f\n", i, error);
    }
    fprintf(gp, "e\n");
    fflush(gp);
    pclose(gp);
    printf("Plot saved as reconstruction_error.png\n");
    return 0;
}

