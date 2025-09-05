#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <jpeglib.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <jpeglib.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>

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
double SVD_reconstruct_topk(const char *filename, int N, int k) {
if (k > N) k = N; 
gsl_matrix *A = imageread_grayscale(filename, N);
gsl_matrix *A_copy = gsl_matrix_alloc(N, N);
gsl_matrix_memcpy(A_copy, A);
gsl_matrix *AtA = gsl_matrix_calloc(N, N);
gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, A, A, 0.0, AtA); //covariance mat
gsl_vector *eval = gsl_vector_alloc(N);
gsl_matrix *evec = gsl_matrix_alloc(N, N);
gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(N);
gsl_eigen_symmv(AtA, eval, evec, w);
gsl_eigen_symmv_free(w);
gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_VAL_DESC);//eigen decompose
gsl_matrix *Vk = gsl_matrix_alloc(N, k);      
gsl_matrix *Sigma_k = gsl_matrix_calloc(k, k); //diag mat 
for (int j = 0; j < k; j++) {
double lambda = gsl_vector_get(eval, j);
double sigma = sqrt(lambda);
gsl_matrix_set(Sigma_k, j, j, sigma);
for (int i = 0; i < N; i++) {
gsl_matrix_set(Vk, i, j, gsl_matrix_get(evec, i, j));// right singular matrix
}}
gsl_matrix *Sigma_inv = gsl_matrix_calloc(k, k);
for (int i = 0; i < k; i++) {
double val = gsl_matrix_get(Sigma_k, i, i);
gsl_matrix_set(Sigma_inv, i, i, (val > 1e-10) ? 1.0 / val : 0.0);
}
gsl_matrix *AV = gsl_matrix_calloc(N, k);
gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, Vk, 0.0, AV);
gsl_matrix *Uk = gsl_matrix_calloc(N, k);
gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, AV, Sigma_inv, 0.0, Uk);//left singualr matrix
gsl_matrix *VkT = gsl_matrix_calloc(k, N);
for (int i = 0; i < N; i++)
for (int j = 0; j < k; j++)
gsl_matrix_set(VkT, j, i, gsl_matrix_get(Vk, i, j));
gsl_matrix *US = gsl_matrix_calloc(N, k);
gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Uk, Sigma_k, 0.0, US);
gsl_matrix *Ak = gsl_matrix_calloc(N, N);
gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, US, VkT, 0.0, Ak);
double err = 0.0;
for (int i = 0; i < N; i++)
for (int j = 0; j < N; j++)
err += fabs(gsl_matrix_get(A_copy, i, j) - gsl_matrix_get(Ak, i, j));
gsl_matrix_free(A);
gsl_matrix_free(A_copy);
gsl_matrix_free(AtA);
gsl_vector_free(eval);
gsl_matrix_free(evec);
gsl_matrix_free(Vk);
gsl_matrix_free(Sigma_k);
gsl_matrix_free(Sigma_inv);
gsl_matrix_free(AV);
gsl_matrix_free(Uk);
gsl_matrix_free(VkT);
gsl_matrix_free(US);
gsl_matrix_free(Ak);
return err;
}
int main(int argc, char *argv[]) {
const char *filename = argv[1];
int N = 256;
FILE *gp = popen("gnuplot", "w");
fprintf(gp, "set terminal pngcairo size 800,600 enhanced font 'Arial,12'\n");
fprintf(gp, "set output 'reconstruction_error.png'\n");
fprintf(gp, "set title 'Reconstruction Error vs k'\n");
fprintf(gp, "set xlabel 'k (Number of singular values)'\n");
fprintf(gp, "set ylabel 'Relative Frobenius Error'\n");
fprintf(gp, "set grid\n");
fprintf(gp, "plot '-' with linespoints title 'Error'\n");
for (int i = 1; i <= N; i += 2) {
double rel_err = SVD_reconstruct_topk(filename, N, i);
fprintf(gp, "%d %f\n", i, rel_err);
printf("k = %d, Relative Error = %.6f\n", i, rel_err);
}
fprintf(gp, "e\n");
fflush(gp);
pclose(gp);
printf("Plot saved as reconstruction_error.png\n");
return 0;
}

