#include <CL/sycl.hpp>
#include <iostream>

#define GROUP_SIZE 16 //16x16 blocks
#define SUB_GROUP_SIZE 16
#define GROUP_COUNT 32
#define THREAD_COUNT (GROUP_SIZE * GROUP_COUNT)
#define SUB_GROUP_COUNT (THREAD_COUNT / SUB_GROUP_SIZE)

void matrix_vector_multiplication(sycl::queue q, std::vector<float> &matrix, std::vector<float> &vec, std::vector<float> &res)
{
	std::vector<float> matrix_res(SUB_GROUP_COUNT * THREAD_COUNT);
	try
	{
		sycl::buffer<float, 1> matrix_buf(matrix.data(), matrix.size());
		sycl::buffer<float, 1> vec_buf(vec.data(), vec.size());
		sycl::buffer<float, 1> res_buf(res.data(), res.size());
		sycl::buffer<float, 1> matrix_res_buf(matrix_res.data(), matrix_res.size());
		q.submit([&](sycl::handler &cgh)
		{
			auto matrix_acc = matrix_buf.get_access<sycl::access::mode::read>(cgh);
			auto vec_acc = vec_buf.get_access<sycl::access::mode::read>(cgh);
			auto res_acc = res_buf.get_access<sycl::access::mode::write>(cgh);
			auto matrix_res_acc = matrix_res_buf.get_access<sycl::access::mode::write>(cgh);
			cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(THREAD_COUNT, THREAD_COUNT), sycl::range<2>(GROUP_SIZE, GROUP_SIZE)),
				[=](sycl::nd_item<2> item)[[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]]
			{
				size_t i = item.get_global_id(0);
				size_t j = item.get_global_id(1);
				float val = matrix_acc[i * THREAD_COUNT + j] * vec_acc[j];
				float sum = sycl::reduce_over_group(item.get_sub_group(), val, std::plus<float>());

				if (item.get_sub_group().get_local_id().get(0) == 0)
				{
					matrix_res_acc[i * SUB_GROUP_COUNT + j / SUB_GROUP_COUNT] = sum;
				}
				item.barrier();

				for (size_t t = 0; t < SUB_GROUP_COUNT; ++t)
				{
					res_acc[i] += matrix_res_acc[i * SUB_GROUP_COUNT + t];
				}
			});
		}).wait_and_throw();
	}
	catch (sycl::exception &e)
	{
		std::cout << e.what() << std::endl;
	}
}

int main() {
	std::vector<float> matrix(THREAD_COUNT * THREAD_COUNT, 2.f);
	std::vector<float> vec(THREAD_COUNT, 3.f);
	std::vector<float> res(THREAD_COUNT, 0.f);
	sycl::queue Q(sycl::host_selector{});
	matrix_vector_multiplication(Q, matrix, vec, res);
	bool correctness = true;
	for (size_t i = 0; i < THREAD_COUNT; ++i)
	{
		if (res[i] != 3072.f)
		{
			correctness = false;
			std::cout << i << ' ' << res[i] << std::endl;
		}
	}
	std::cout << correctness << std::endl;
	return 0;
}
