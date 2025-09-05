Perfect! We can add a **Run Instructions** section to the README with your exact commands. Here’s the updated **README.md** including that:

````markdown
# EVD_and_SVD

This repository contains C implementations for **Eigenvalue Decomposition (EVD)** and **Singular Value Decomposition (SVD)** for grayscale images using the [GSL (GNU Scientific Library)](https://www.gnu.org/software/gsl/) and [libjpeg](http://libjpeg.sourceforge.net/) libraries. The programs also generate plots of reconstruction error versus the number of eigenvalues/singular values used.

---

## Files

- **EVD.c**:  
  Computes the top-`k` eigenvalues and eigenvectors of a grayscale image, reconstructs the image using them, and calculates the reconstruction error. It also generates a plot (`reconstruction_error.png`) showing error versus `k`.

- **SVD.c**:  
  Computes the top-`k` singular values and corresponding singular vectors of a grayscale image, reconstructs the image using them, and calculates the relative Frobenius error. It also generates a plot (`reconstruction_error.png`) showing error versus `k`.

---

## Requirements

- **C Compiler**: `gcc` or similar
- **Libraries**:
  - [GSL](https://www.gnu.org/software/gsl/)
  - [libjpeg](http://libjpeg.sourceforge.net/)
  - `gnuplot` (for generating plots)

On Ubuntu/Debian, you can install dependencies with:

```bash
sudo apt-get update
sudo apt-get install build-essential libgsl-dev libjpeg-dev gnuplot
````

---

## Compilation

Compile the programs using `gcc`:

```bash
gcc -o EVD EVD.c -lgsl -lgslcblas -lm -ljpeg
gcc -o SVD SVD.c -lgsl -lgslcblas -lm -ljpeg
```

---

## Run Instructions

Run the programs with a grayscale JPEG image as input:

```bash
./SVD "5.jpg"
./EVD "5.jpg"
```

Or, if you prefer explicit compilation and running:

```bash
gcc svd.c -o svd -lgsl -lgslcblas -ljpeg -lm
./svd "5.jpg"

gcc evd.c -o evd -lgsl -lgslcblas -ljpeg -lm
./evd "5.jpg"
```

* The programs print the reconstruction error for different values of `k`.
* A plot `reconstruction_error.png` will be generated showing the error versus `k`.

---

## Notes

* The programs assume square grayscale images of size `256x256` by default (`N = 256`). You can modify `N` in the code if your images have a different size.
* Reconstruction uses top-`k` eigenvalues (for EVD) or singular values (for SVD) to approximate the original image.
* Errors are computed as:

  * **EVD**: sum of absolute differences between original and reconstructed images.
  * **SVD**: relative Frobenius norm error.

---

## References

* [GNU Scientific Library (GSL)](https://www.gnu.org/software/gsl/)
* [libjpeg](http://libjpeg.sourceforge.net/)
* Eigenvalue and Singular Value Decomposition theory

```

This now clearly tells anyone exactly how to **compile and run** both programs.  

If you want, I can also **add an example output screenshot** section so your README looks more professional on GitHub. Do you want me to do that?
```

