#include <CL/sycl.hpp>
#include <iostream>
#include <mkl_dfti.h>
#include <complex>
#define GROUP_SIZE 256
#define GROUP_COUNT 32
#define THREAD_COUNT (GROUP_SIZE * GROUP_COUNT)

const float Pi = 3.14159265359;
typedef std::vector<std::complex<float> > t_complex_vector;
typedef float el_type;

void mkl_fft(t_complex_vector& in, t_complex_vector &out)
{
	DFTI_DESCRIPTOR_HANDLE descriptor;
	MKL_LONG status;
	DFTI_CONFIG_VALUE precision;
	DFTI_CONFIG_VALUE placement;
	precision = DFTI_SINGLE;
	if (&in == &out)
	{
		placement = DFTI_INPLACE;
	}
	else if (&in != &out)
	{
		placement = DFTI_NOT_INPLACE;
	}
	status = DftiCreateDescriptor(&descriptor, precision, DFTI_COMPLEX, 1, in.size()); //Specify size and precision
	status = DftiSetValue(descriptor, DFTI_PLACEMENT, placement); //In/out of place FFT
	status = DftiCommitDescriptor(descriptor); //Finalize the descriptor
	status = DftiComputeForward(descriptor, in.data(), out.data()); //Compute the Forward FFT
	status = DftiFreeDescriptor(&descriptor); //Free the descriptor
}

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

void my_fft_heterogenous(sycl::queue q, size_t size, float *src_real, float *src_imag) //number of threads = number of butterflies
{
	int bit_length = log2(size);
	int iterations = bit_length;
	for (int i = 1; i < size - 1; ++i) //pre permutation algorithm
	{
		size_t j = m_reverse(i, bit_length);
		if (j <= i) continue;
		std::swap(src_real[i], src_real[j]);
		std::swap(src_imag[i], src_imag[j]);
	}
	try
	{
		sycl::buffer<float, 1> real_buf(src_real, size);
		sycl::buffer<float, 1> imag_buf(src_imag, size);
		q.submit([&](sycl::handler &cgh)
		{
			auto real = real_buf.get_access<sycl::access::mode::read_write>(cgh);
			auto imag = imag_buf.get_access<sycl::access::mode::read_write>(cgh);
			cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(size / 2), sycl::range<1>(GROUP_SIZE)),
				[=](sycl::nd_item<1> item)
			{
				for (int i = 0; i < iterations; ++i)
				{
					size_t stride = 1 << i;
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
					
					//wait for every thread to finish within one iteration
					item.barrier(); 
				}
			});
		}).wait_and_throw();
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
	float eps = 0.001;
	size_t size = 8;
	std::cin >> size;
	float* real = new float[size];
	float* imag = new float[size];
	t_complex_vector correct_output(size);
	for (size_t i = 0; i < size; i++)
	{
		real[i] = i;
		imag[i] = i;
		correct_output[i] = std::complex<float>(real[i], imag[i]);
	}
	sycl::queue Q(sycl::host_selector{});
	my_fft_heterogenous(Q, size, real, imag);
	mkl_fft(correct_output, correct_output);
	bool flag = true;
	for (size_t i = 0; i < size; i++)
	{
		if (abs(correct_output[i].real() - real[i]) > eps || abs(correct_output[i].imag() - imag[i]) > eps)
		{
			std::cout << correct_output[i] << '\t' << real[i] << ' ' << imag[i] << std::endl;
			flag = false;
		}
	}
	std::cout << flag;
	return 0;
}
