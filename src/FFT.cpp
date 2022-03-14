#include <CL/sycl.hpp>
#include <iostream>
#include <mkl_dfti.h>
#include <complex>
#include <omp.h>

#define GROUP_SIZE 256
#define GROUP_COUNT 32
#define THREAD_COUNT (GROUP_SIZE * GROUP_COUNT)

const float Pi = 3.14159265359;
typedef std::vector<std::complex<float> > t_complex_vector;
typedef float el_type;

template <typename T>
inline T m_reverse(T a, int bit_len)
{
	T res = 0;
	for (int i = 0; i < bit_len; ++i)
	{
		bool bit = (a >> i) % 2;
		res |= bit << (bit_len - i - 1);
	}
	return res;
}

void m_FFT_vectorized(el_type *src_real, el_type *src_im, el_type *res_real, el_type *res_im, const size_t size, bool mthread_param)
{
	size_t global_subsequence_size = 1;
	const size_t bit_length = log2(size);
	const int iterations = log2(size);
	if (src_real != res_real)
		res_real = src_real;
	if (src_im != res_im)
		res_im = src_im;
#pragma omp parallel for schedule(static) if(mthread_param == 1)
	for (size_t i = 1; i < size - 1; ++i)
	{
		size_t j = m_reverse(i, bit_length);
		if (j <= i) continue;
		std::swap(res_real[i], res_real[j]);
		std::swap(res_im[i], res_im[j]);
	}

	size_t subsequence_size = global_subsequence_size;
	for (size_t i = 0; i < iterations; ++i)
	{
#pragma omp parallel for schedule(static) if(mthread_param == 1)
		for (int j = 0; j < size / (subsequence_size * 2); ++j)
		{
#pragma omp simd
			for (int t = 0; t < subsequence_size; ++t)
			{
				size_t t_adress = j * subsequence_size * 2 + t;
				el_type temp_cos = cos(Pi / subsequence_size * t);
				el_type temp_sin = -sin(Pi / subsequence_size * t);
				el_type temp_first = (temp_cos * res_real[t_adress + subsequence_size] - temp_sin * res_im[t_adress + subsequence_size]);
				el_type temp_second = (temp_sin * res_real[t_adress + subsequence_size] + temp_cos * res_im[t_adress + subsequence_size]);
				el_type temp_real_t = res_real[t_adress] + temp_first;
				el_type temp_imag_t = res_im[t_adress] + temp_second;
				el_type temp_real_ss_plus_t = res_real[t_adress] - temp_first;
				el_type temp_imag_ss_plus_t = res_im[t_adress] - temp_second;
				res_real[t_adress] = temp_real_t;
				res_im[t_adress] = temp_imag_t;
				res_real[t_adress + subsequence_size] = temp_real_ss_plus_t;
				res_im[t_adress + subsequence_size] = temp_imag_ss_plus_t;
			}
		}
		subsequence_size *= 2;
	}
}

//number of threads = number of butterflies
double my_fft_heterogeneous(sycl::queue& q, size_t size, float *src_real, float *src_imag) 
{
	int bit_length = log2(size);
	int iterations = bit_length;
	double elapsed_time = 0;
	for (size_t i = 1; i < size - 1; ++i) //pre permutation algorithm
	{
		size_t j = m_reverse(i, bit_length);
		if (j <= i) continue;
		std::swap(src_real[i], src_real[j]);
		std::swap(src_imag[i], src_imag[j]);
	}
	try
	{
		//sycl::buffer<float, 1> real_buf(src_real, size);
		//sycl::buffer<float, 1> imag_buf(src_imag, size);
		float* real = sycl::malloc_device<float>(size, q);
		float* imag = sycl::malloc_device<float>(size, q);
		q.submit([&](sycl::handler &cgh)
		{
			q.memcpy(real, src_real, size * sizeof(float));
		}).wait_and_throw();
		q.submit([&](sycl::handler &cgh)
		{
			q.memcpy(imag, src_imag, size * sizeof(float));
		}).wait_and_throw();
		for (size_t i = 0; i < iterations; ++i)
		{
			auto ev = q.submit([&](sycl::handler &cgh)
			{
				//auto real = real_buf.get_access<sycl::access::mode::read_write>(cgh);
				//auto imag = imag_buf.get_access<sycl::access::mode::read_write>(cgh);

				cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(size / 2), sycl::range<1>(GROUP_SIZE)),
					[=](sycl::nd_item<1> item)
				{
					size_t stride = 1ull << i;
					size_t first_adress = item.get_global_id(0) % stride + (item.get_global_id(0) / stride * 2 * stride);
					size_t t = item.get_global_id(0) % stride;
					size_t second_adress = first_adress + stride;
					float temp_cos = cos(Pi / stride * t);
					float temp_sin = -sin(Pi / stride * t);
					float temp_first = (temp_cos * real[second_adress] - temp_sin * imag[second_adress]);
					float temp_second = (temp_sin * real[second_adress] + temp_cos * imag[second_adress]);
					float temp_real_first_adress = real[first_adress] + temp_first;
					float temp_imag_first_adress = imag[first_adress] + temp_second;
					float temp_real_second_adress = real[first_adress] - temp_first;
					float temp_imag_second_adress = imag[first_adress] - temp_second;
					real[first_adress] = temp_real_first_adress;
					imag[first_adress] = temp_imag_first_adress;
					real[second_adress] = temp_real_second_adress;
					imag[second_adress] = temp_imag_second_adress;
				});

			});
			ev.wait_and_throw();
			elapsed_time += ((ev.get_profiling_info<sycl::info::event_profiling::command_end>() - ev.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1.0e9);
		}
		q.submit([&](sycl::handler &cgh)
		{
			q.memcpy(src_real, real, size * sizeof(float));
		}).wait_and_throw();
		q.submit([&](sycl::handler &cgh)
		{
			q.memcpy(src_imag, imag, size * sizeof(float));
		}).wait_and_throw();
		//sycl::free(real, q);
		//sycl::free(imag, q);
		return elapsed_time;
	}
	catch (sycl::exception &e)
	{
		std::cout << e.what();
	}
	catch (std::exception &e)
	{
		std::cout << e.what();
	}
}

int main() {
	float eps = 0.1;
	size_t size = 1ull << 25;
	double ffth_start = 0,ffth_time = 0, fftv_start = 0, fftv_time = 0;
	float* real = static_cast<float *>(malloc(size * sizeof(float)));
	float* imag = static_cast<float *>(malloc(size * sizeof(float)));
	float* correct_real = new float[size];
	float* correct_imag = new float[size];
	for (size_t i = 0; i < size; i++)
	{
		correct_real[i] = real[i] = rand() % 2000 / 1000.f - 1;
		correct_imag[i] = imag[i] = rand() % 2000 / 1000.f - 1;
	}
	sycl::queue Q(sycl::default_selector{}, sycl::property::queue::enable_profiling());
	ffth_start = omp_get_wtime();
	my_fft_heterogeneous(Q, size, real, imag);
	ffth_time = omp_get_wtime() - ffth_start;
	fftv_start = omp_get_wtime();
	m_FFT_vectorized(correct_real, correct_imag, correct_real, correct_imag, size, 1);
	fftv_time = omp_get_wtime() - fftv_start;
	bool flag = true;
	int max_output_count = 20;
	for (size_t i = 0; i < size; i++)
	{
		if (abs(correct_real[i] - real[i]) > eps || abs(correct_imag[i] - imag[i]) > eps)
		{
			if (max_output_count > 0)
			{
				std::cout << i << ' ' << correct_real[i] << ' ' << correct_imag[i] << '\t' << real[i] << ' ' << imag[i] << std::endl;
				max_output_count--;
			}
			flag = false;
		}
	}
	std::cout.precision(3);
	std::cout << flag << std::endl << "OMP: " << fftv_time << std::endl << "DPC++ " << ffth_time << std::endl;
	return 0;
}
