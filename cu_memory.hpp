/**
	\file		cu_memory.hpp
	\brief		Memory functions
	\author		Seongoh Lee
	\version	1.0
	\date		2011.12.09
*/

#ifndef _CU_MEMORY_HPP
#define _CU_MEMORY_HPP

#include "cu_device.h"
#include <vector>

//template<class T>
//class cu_array {
//public:
//	typedef T type;
//	enum { elem_size = sizeof(T) };
//	void malloc( T **d_ptr, size_t size ) { cu_malloc( (void**)d_ptr, size * elem_size ); }
//	void free( T *d_ptr ) { cu_free( d_ptr ); }
//	void memcpy_device_to_host( T *dst, T *src, size_t size ) { cu_memcpy_device_to_host( dst, src, size * elem_size ); }
//	void memcpy_host_to_device( T *dst, T *src, size_t size ) { cu_memcpy_host_to_device( dst, src, size * elem_size ); }
//	void memcpy_device_to_device( T *dst, T *src, size_t size ) { cu_memcpy_device_to_device( dst, src, size * elem_size ); }
//};
//
//template<class T>
//class cu_array_2d {
//public:
//	typedef T type;
//	enum { elem_size = sizeof(T) };
//	void malloc( T **d_ptr, size_t *pitch, size_t width, size_t height ) { cu_malloc_2d( (void**)d_ptr, pitch, width * elem_size, height ); }
//	void free( T *d_ptr ) { cu_free( d_ptr ); }
//	void memcpy_device_to_host( T *dst, size_t dpitch, T *src, size_t spitch, size_t width, size_t height )
//	{ cu_memcpy_2d_device_to_host( dst, dpitch, src, spitch, width * elem_size, height ); }
//	void memcpy_host_to_device( T *dst, size_t dpitch, T *src, size_t spitch, size_t width, size_t height )
//	{ cu_memcpy_2d_host_to_device( dst, dpitch, src, spitch, width * elem_size, height ); }
//	void memcpy_device_to_device( T *dst, size_t dpitch, T *src, size_t spitch, size_t width, size_t height )
//	{ cu_memcpy_2d_device_to_device( dst, dpitch, src, spitch, width * elem_size, height ); }
//};

//////////////////////
// Kernel container //
//////////////////////

template<typename T> struct cu_ptr
{
	typedef T elem_type;
	const static size_t elem_size_ = sizeof(elem_type);

	T* data;

	cuda_function_header cu_ptr() : data(0) {}
	cuda_function_header cu_ptr(T* data_arg) : data(data_arg) {}

	cuda_function_header size_t elem_size() const { return elem_size_; }
	cuda_function_header operator       T*()       { return data; }
	cuda_function_header operator const T*() const { return data; }
};

template<typename T> struct ptr_size : public cu_ptr<T>
{                     
	cuda_function_header ptr_size() : size(0) {}
	cuda_function_header ptr_size(T* data_arg, size_t size_arg) : cu_ptr<T>(data_arg), size(size_arg) {}

	size_t size;
};

template<typename T>  struct ptr_step : public cu_ptr<T>
{   
	cuda_function_header ptr_step() : step(0) {}
	cuda_function_header ptr_step(T* data_arg, size_t step_arg) : cu_ptr<T>(data_arg), step(step_arg) {}

	/** \brief stride between two consecutive rows in bytes. Step is stored always and everywhere in bytes!!! */
	size_t step;

	cuda_function_header		T* ptr(int y = 0)       { return (      T*)( (      char*)cu_ptr<T>::data + y * step); }
	cuda_function_header const	T* ptr(int y = 0) const { return (const T*)( (const char*)cu_ptr<T>::data + y * step); }
};

template <typename T> struct ptr_step_size : public ptr_step<T>
{   
	cuda_function_header ptr_step_size() : cols(0), rows(0) {}
	cuda_function_header ptr_step_size(int rows_arg, int cols_arg, T* data_arg, size_t step_arg) 
		: ptr_step<T>(data_arg, step_arg), cols(cols_arg), rows(rows_arg) {}

	int cols;
	int rows;                                                                              
};

///////////////
// cu_memory //
///////////////

class cu_memory
{
public:
	/** \brief Empty constructor. */
	cu_memory();

	/** \brief Destructor. */
	~cu_memory();            

	/** \brief Allocates internal buffer in GPU memory
	* \param size_bytes_arg: amount of memory to allocate
	* */
	cu_memory(size_t size_bytes_arg);

	/** \brief Initializes with user allocated buffer. Reference counting is disabled in this case.
	* \param ptr_arg: pointer to buffer
	* \param size_bytes_arg: buffer size
	* */
	cu_memory(void *ptr_arg, size_t size_bytes_arg);

	/** \brief Copy constructor. Just increments reference counter. */
	cu_memory(const cu_memory& other_arg);

	/** \brief Assigment operator. Just increments reference counter. */
	cu_memory& operator=(const cu_memory& other_arg);

	/** \brief Allocates internal buffer in GPU memory. If internal buffer was created before the function recreates it with new size. If new and old sizes are equal it does nothing.               
	* \param size_bytes_arg: buffer size
	* */
	void create(size_t size_bytes_arg);

	/** \brief Decrements reference counter and releases internal buffer if needed. */
	void release();

	/** \brief Performs data copying. If destination size differs it will be reallocated.
	* \param other_arg: destination container
	* */
	void copy_to(cu_memory& other) const;

	/** \brief Uploads data to internal buffer in GPU memory. It calls create() inside to ensure that intenal buffer size is enough.
	* \param host_ptr_arg: pointer to buffer to upload               
	* \param size_bytes_arg: buffer size
	* */
	void upload(const void *host_ptr_arg, size_t size_bytes_arg);

	/** \brief Downloads data from internal buffer to CPU memory
	* \param host_ptr_arg: pointer to buffer to download               
	* */
	void download(void *host_ptr_arg) const;

	/** \brief Performs swap of data pointed with another device memory. 
	* \param other: device memory to swap with   
	* */
	void swap(cu_memory& other_arg);

	/** \brief Returns pointer for internal buffer in GPU memory. */
	template<class T> T* ptr();

	/** \brief Returns constant pointer for internal buffer in GPU memory. */            
	template<class T> const T* ptr() const;

	/** \brief Conversion to ptr_size for passing to kernel functions. */
	template <class U> operator ptr_size<U>() const;            

	/** \brief Returns true if unallocated otherwise false. */
	bool empty() const;

	size_t size_bytes() const;

private:
	/** \brief Device pointer. */
	void *data_;

	/** \brief Allocated size in bytes. */
	size_t size_bytes_;

	/** \brief Pointer to reference counter in CPU memory. */
	int* refcount_;
};

////////////////////
// cu_memory_host //
////////////////////

class cu_memory_host
{
public:
	/** \brief Empty constructor. */
	cu_memory_host();

	/** \brief Destructor. */
	~cu_memory_host();            

	/** \brief Allocates internal buffer in CPU memory
	* \param size_bytes_arg: amount of memory to allocate
	* */
	cu_memory_host(size_t size_bytes_arg);

	/** \brief Initializes with user allocated buffer. Reference counting is disabled in this case.
	* \param ptr_arg: pointer to buffer
	* \param size_bytes_arg: buffer size
	* */
	cu_memory_host(void *ptr_arg, size_t size_bytes_arg);

	/** \brief Copy constructor. Just increments reference counter. */
	cu_memory_host(const cu_memory_host& other_arg);

	/** \brief Assigment operator. Just increments reference counter. */
	cu_memory_host& operator=(const cu_memory_host& other_arg);

	/** \brief Allocates internal buffer in CPU memory. If internal buffer was created before the function recreates it with new size. If new and old sizes are equal it does nothing.               
	* \param size_bytes_arg: buffer size
	* */
	void create(size_t size_bytes_arg);

	/** \brief Decrements reference counter and releases internal buffer if needed. */
	void release();

	/** \brief Performs data copying. If destination size differs it will be reallocated.
	* \param other_arg: destination container
	* */
	void copy_to(cu_memory_host& other) const;

	/** \brief Uploads data to internal buffer in CPU memory. It calls create() inside to ensure that intenal buffer size is enough.
	* \param host_ptr_arg: pointer to buffer to upload               
	* \param size_bytes_arg: buffer size
	* */
	void upload(const void *device_ptr_arg, size_t size_bytes_arg);

	/** \brief Downloads data from internal buffer to GPU memory
	* \param host_ptr_arg: pointer to buffer to download               
	* */
	void download(void *device_ptr_arg) const;

	/** \brief Performs swap of data pointed with another device memory. 
	* \param other: device memory to swap with   
	* */
	void swap(cu_memory_host& other_arg);

	/** \brief Returns pointer for internal buffer in CPU memory. */
	template<class T> T* ptr();

	/** \brief Returns constant pointer for internal buffer in CPU memory. */            
	template<class T> const T* ptr() const;

	/** \brief Conversion to ptr_size for passing to kernel functions. */
	template <class U> operator ptr_size<U>() const;            

	/** \brief Returns true if unallocated otherwise false. */
	bool empty() const;

	size_t size_bytes() const;

private:
	/** \brief Device pointer. */
	void *data_;

	/** \brief Allocated size in bytes. */
	size_t size_bytes_;

	/** \brief Pointer to reference counter in CPU memory. */
	int* refcount_;
};

//////////////////
// cu_memory_2d //
//////////////////

class cu_memory_2d
{
public:
	/** \brief Empty constructor. */
	cu_memory_2d();

	/** \brief Destructor. */
	~cu_memory_2d();            

	/** \brief Allocates internal buffer in memory
	* \param rows_arg: number of rows to allocate
	* \param colsBytes_arg: width of the buffer in bytes
	* */
	cu_memory_2d(int rows_arg, int colsBytes_arg);

	/** \brief Initializes with user allocated buffer. Reference counting is disabled in this case.
	* \param rows_arg: number of rows
	* \param colsBytes_arg: width of the buffer in bytes
	* \param data_arg: pointer to buffer
	* \param stepBytes_arg: stride between two consecutive rows in bytes
	* */
	cu_memory_2d(int rows_arg, int colsBytes_arg, void *data_arg, size_t step_arg);

	/** \brief Copy constructor. Just increments reference counter. */
	cu_memory_2d(const cu_memory_2d& other_arg);

	/** \brief Assigment operator. Just increments reference counter. */
	cu_memory_2d& operator=(const cu_memory_2d& other_arg);

	/** \brief Allocates internal buffer in GPU memory. If internal buffer was created before the function recreates it with new size. If new and old sizes are equal it does nothing.
	* \param ptr_arg: number of rows to allocate
	* \param sizeBytes_arg: width of the buffer in bytes
	* */
	void create(int rows_arg, int colsBytes_arg);

	/** \brief Decrements reference counter and releases internal buffer if needed. */
	void release();

	/** \brief Performs data copying. If destination size differs it will be reallocated.
	* \param other_arg: destination container
	* */
	void copy_to(cu_memory_2d& other) const;

	//** \brief Uploads data to internal buffer in GPU memory. It calls create() inside to ensure that intenal buffer size is enough.
	//* \param cu_ptr_arg: pointer to host buffer to upload               
	//* \param host_step_arg: stride between two consecutive rows in bytes for host buffer
	//* \param rows_arg: number of rows to upload
	//* \param sizeBytes_arg: width of host buffer in bytes
	//* */
	void upload(const void *cu_ptr_arg, size_t host_step_arg, int rows_arg, int colsBytes_arg);

	//** \brief Downloads data from internal buffer to CPU memory. User is resposible for correct host buffer size.
	//* \param cu_ptr_arg: pointer to host buffer to download               
	//* \param host_step_arg: stride between two consecutive rows in bytes for host buffer             
	//* */
	void download(void *cu_ptr_arg, size_t host_step_arg) const;

	/** \brief Performs swap of data pointed with another device memory. 
	* \param other: device memory to swap with   
	* */
	void swap(cu_memory_2d& other_arg);

	/** \brief Returns pointer to given row in internal buffer. 
	* \param y_arg: row index   
	* */
	template<class T> T* ptr(int y_arg = 0);

	/** \brief Returns constant pointer to given row in internal buffer. 
	* \param y_arg: row index   
	* */
	template<class T> const T* ptr(int y_arg = 0) const;

	/** \brief Conversion to PtrStep for passing to kernel functions. */
	template <class U> operator ptr_step<U>() const;            

	/** \brief Conversion to PtrStepSz for passing to kernel functions. */
	template <class U> operator ptr_step_size<U>() const;

	/** \brief Returns true if unallocated otherwise false. */
	bool empty() const;

	/** \brief Returns number of bytes in each row. */
	int cols_bytes() const;

	/** \brief Returns number of rows. */
	int rows() const;

	/** \brief Returns stride between two consecutive rows in bytes for internal buffer. Step is stored always and everywhere in bytes!!! */
	size_t step() const;                               
private:
	/** \brief Device pointer. */
	void *data_;

	/** \brief Stride between two consecutive rows in bytes for internal buffer. Step is stored always and everywhere in bytes!!! */
	size_t step_;

	/** \brief Width of the buffer in bytes. */
	int cols_bytes_;

	/** \brief Number of rows. */
	int rows_;

	/** \brief Pointer to reference counter in CPU memory. */
	int* refcount_;
};

//////////////
// cu_array //
//////////////

template<class T> 
class cu_array : public cu_memory
{
public:
	/** \brief Element type. */
	typedef T type;

	/** \brief Element size. */
	enum { elem_size = sizeof(T) };

	/** \brief Empty constructor. */
	cu_array();

	/** \brief Allocates internal buffer in GPU memory
	* \param size_t: number of elements to allocate
	* */
	cu_array(size_t size);

	/** \brief Initializes with user allocated buffer. Reference counting is disabled in this case.
	* \param ptr: pointer to buffer
	* \param size: elemens number
	* */
	cu_array(T *ptr, size_t size);

	/** \brief Copy constructor. Just increments reference counter. */
	cu_array(const cu_array& other);

	/** \brief Assigment operator. Just increments reference counter. */
	cu_array& operator = (const cu_array& other);

	/** \brief Allocates internal buffer in GPU memory. If internal buffer was created before the function recreates it with new size. If new and old sizes are equal it does nothing.               
	* \param size: elemens number
	* */
	void create(size_t size);

	/** \brief Decrements reference counter and releases internal buffer if needed. */
	void release();  

	/** \brief Performs data copying. If destination size differs it will be reallocated.
	* \param other_arg: destination container
	* */
	void copy_to(cu_array& other) const;

	/** \brief Uploads data to internal buffer in GPU memory. It calls create() inside to ensure that intenal buffer size is enough.
	* \param host_ptr_arg: pointer to buffer to upload               
	* \param size: elemens number
	* */
	void upload(const T *host_ptr, size_t size);

	/** \brief Downloads data from internal buffer to CPU memory
	* \param host_ptr_arg: pointer to buffer to download               
	* */
	void download(T *host_ptr) const;

	/** \brief Uploads data to internal buffer in GPU memory. It calls create() inside to ensure that intenal buffer size is enough.
	* \param data: host vector to upload from              
	* */
	template<class A>
	void upload(const std::vector<T, A>& data);

	/** \brief Downloads data from internal buffer to CPU memory
	* \param data:  host vector to download to                 
	* */
	template<typename A>
	void download(std::vector<T, A>& data) const;

	/** \brief Performs swap of data pointed with another device array. 
	* \param other: device array to swap with   
	* */
	void swap(cu_array& other_arg);

	/** \brief Returns pointer for internal buffer in GPU memory. */
	T* ptr(); 

	/** \brief Returns const pointer for internal buffer in GPU memory. */
	const T* ptr() const;

	//using cu_memory::ptr;

	/** \brief Returns pointer for internal buffer in GPU memory. */
	operator T*();

	/** \brief Returns const pointer for internal buffer in GPU memory. */
	operator const T*() const;

	/** \brief Returns size in elements. */
	size_t size() const;            
};

///////////////////
// cu_array_host //
///////////////////

template<class T> 
class cu_array_host : public cu_memory_host
{
public:
	/** \brief Element type. */
	typedef T type;

	/** \brief Element size. */
	enum { elem_size = sizeof(T) };

	/** \brief Empty constructor. */
	cu_array_host();

	/** \brief Allocates internal buffer in GPU memory
	* \param size_t: number of elements to allocate
	* */
	cu_array_host(size_t size);

	/** \brief Initializes with user allocated buffer. Reference counting is disabled in this case.
	* \param ptr: pointer to buffer
	* \param size: elemens number
	* */
	cu_array_host(T *ptr, size_t size);

	/** \brief Copy constructor. Just increments reference counter. */
	cu_array_host(const cu_array_host& other);

	/** \brief Assigment operator. Just increments reference counter. */
	cu_array_host& operator = (const cu_array_host& other);

	/** \brief Allocates internal buffer in GPU memory. If internal buffer was created before the function recreates it with new size. If new and old sizes are equal it does nothing.               
	* \param size: elemens number
	* */
	void create(size_t size);

	/** \brief Decrements reference counter and releases internal buffer if needed. */
	void release();  

	/** \brief Performs data copying. If destination size differs it will be reallocated.
	* \param other_arg: destination container
	* */
	void copy_to(cu_array_host& other) const;

	/** \brief Uploads data to internal buffer in CPU memory. It calls create() inside to ensure that intenal buffer size is enough.
	* \param host_ptr_arg: pointer to buffer to upload               
	* \param size: elemens number
	* */
	void upload(const T *host_ptr, size_t size);

	/** \brief Downloads data from internal buffer to GPU memory
	* \param host_ptr_arg: pointer to buffer to download               
	* */
	void download(T *host_ptr) const;

	/** \brief Uploads data to internal buffer in CPU memory. It calls create() inside to ensure that intenal buffer size is enough.
	* \param data: host vector to upload from              
	* */
	template<class A>
	void upload(const std::vector<T, A>& data);

	/** \brief Downloads data from internal buffer to GPU memory
	* \param data:  host vector to download to                 
	* */
	template<typename A>
	void download(std::vector<T, A>& data) const;

	/** \brief Performs swap of data pointed with another device array. 
	* \param other: device array to swap with   
	* */
	void swap(cu_array_host& other_arg);

	/** \brief Returns pointer for internal buffer in GPU memory. */
	T* ptr(); 

	/** \brief Returns const pointer for internal buffer in GPU memory. */
	const T* ptr() const;

	//using cu_memory::ptr;

	/** \brief Returns pointer for internal buffer in GPU memory. */
	operator T*();

	/** \brief Returns const pointer for internal buffer in GPU memory. */
	operator const T*() const;

	/** \brief Returns size in elements. */
	size_t size() const;            
};

/////////////////
// cu_array_2d //
/////////////////

template<class T> 
class cu_array_2d : public cu_memory_2d
{
public:
	/** \brief Element type. */
	typedef T type;

	/** \brief Element size. */
	enum { elem_size = sizeof(T) };

	/** \brief Empty constructor. */
	cu_array_2d();

	/** \brief Allocates internal buffer in GPU memory
	* \param rows: number of rows to allocate
	* \param cols: number of elements in each row
	* */
	cu_array_2d(int rows, int cols);

	/** \brief Initializes with user allocated buffer. Reference counting is disabled in this case.
	* \param rows: number of rows
	* \param cols: number of elements in each row
	* \param data: pointer to buffer
	* \param step_bytes: stride between two consecutive rows in bytes
	* */
	cu_array_2d(int rows, int cols, void *data, size_t step_bytes_);

	/** \brief Copy constructor. Just increments reference counter. */
	cu_array_2d(const cu_array_2d& other);

	/** \brief Assigment operator. Just increments reference counter. */
	cu_array_2d& operator = (const cu_array_2d& other);

	/** \brief Allocates internal buffer in GPU memory. If internal buffer was created before the function recreates it with new size. If new and old sizes are equal it does nothing.
	* \param rows: number of rows to allocate
	* \param cols: number of elements in each row
	* */
	void create(int rows, int cols);

	/** \brief Decrements reference counter and releases internal buffer if needed. */
	void release();

	/** \brief Performs data copying. If destination size differs it will be reallocated.
	* \param other: destination container
	* */
	void copy_to(cu_array_2d& other) const;

	/** \brief Uploads data to internal buffer in GPU memory. It calls create() inside to ensure that intenal buffer size is enough.
	* \param host_ptr: pointer to host buffer to upload               
	* \param host_step: stride between two consecutive rows in bytes for host buffer
	* \param rows: number of rows to upload
	* \param cols: number of elements in each row
	* */
	void upload(const void *host_ptr, size_t host_step, int rows, int cols);

	/** \brief Downloads data from internal buffer to CPU memory. User is resposible for correct host buffer size.
	* \param host_ptr: pointer to host buffer to download               
	* \param host_step: stride between two consecutive rows in bytes for host buffer             
	* */
	void download(void *host_ptr, size_t host_step) const;

	/** \brief Performs swap of data pointed with another device array. 
	* \param other: device array to swap with   
	* */
	void swap(cu_array_2d& other_arg);

	/** \brief Uploads data to internal buffer in GPU memory. It calls create() inside to ensure that intenal buffer size is enough.
	* \param data: host vector to upload from              
	* \param cols: stride in elements between two consecutive rows for host buffer
	* */
	template<class A>
	void upload(const std::vector<T, A>& data, int cols);

	/** \brief Downloads data from internal buffer to CPU memory
	* \param data: host vector to download to                 
	* \param cols: Output stride in elements between two consecutive rows for host vector.
	* */
	template<class A>
	void download(std::vector<T, A>& data, int& cols) const;

	/** \brief Returns pointer to given row in internal buffer. 
	* \param y_arg: row index   
	* */
	T* ptr(int y = 0);             

	/** \brief Returns const pointer to given row in internal buffer. 
	* \param y_arg: row index   
	* */
	const T* ptr(int y = 0) const;            

	//using cu_memory_2d::ptr;            

	/** \brief Returns pointer for internal buffer in GPU memory. */
	operator T*();

	/** \brief Returns const pointer for internal buffer in GPU memory. */
	operator const T*() const;                        

	/** \brief Returns number of elements in each row. */
	int cols() const;

	/** \brief Returns number of rows. */
	int rows() const;

	/** \brief Returns step in elements. */
	size_t elem_step() const;
};        

/////////////////////  Inline implementations of cu_memory ////////////////////////////////////////////
    
template<class T> inline       T* cu_memory::ptr()       { return (      T*)data_; }
template<class T> inline const T* cu_memory::ptr() const { return (const T*)data_; }
                        
template <class U> inline cu_memory::operator ptr_size<U>() const
{
    ptr_size<U> result;
    result.data = (U*)ptr<U>();
    result.size = size_bytes_/sizeof(U);
    return result; 
}

/////////////////////  Inline implementations of cu_memory_host ///////////////////////////////////////
    
template<class T> inline       T* cu_memory_host::ptr()       { return (      T*)data_; }
template<class T> inline const T* cu_memory_host::ptr() const { return (const T*)data_; }
                        
template <class U> inline cu_memory_host::operator ptr_size<U>() const
{
    ptr_size<U> result;
    result.data = (U*)ptr<U>();
    result.size = size_bytes_/sizeof(U);
    return result; 
}

/////////////////////  Inline implementations of cu_memory_2d ////////////////////////////////////////////
               
template<class T>        T* cu_memory_2d::ptr(int y_arg)       { return (      T*)((      char*)data_ + y_arg * step_); }
template<class T>  const T* cu_memory_2d::ptr(int y_arg) const { return (const T*)((const char*)data_ + y_arg * step_); }
  
template<class U> cu_memory_2d::operator ptr_step<U>() const
{
    ptr_step<U> result;
    result.data = (U*)ptr<U>();
    result.step = step_;
    return result;
}

template<class U> cu_memory_2d::operator ptr_step_size<U>() const
{
    ptr_step_size<U> result;
    result.data = (U*)ptr<U>();
    result.step = step_;
    result.cols = cols_bytes_/sizeof(U);
    result.rows = rows_;
    return result;
}

///////////////////  Inline implementations of cu_array ////////////////////////////////////////////

template<class T> inline cu_array<T>::cu_array() {}
template<class T> inline cu_array<T>::cu_array(size_t size) : cu_memory(size * elem_size) {}
template<class T> inline cu_array<T>::cu_array(T *ptr, size_t size) : cu_memory(ptr, size * elem_size) {}
template<class T> inline cu_array<T>::cu_array(const cu_array& other) : cu_memory(other) {}
template<class T> inline cu_array<T>& cu_array<T>::operator=(const cu_array& other)
{ cu_memory::operator=(other); return *this; }

template<class T> inline void cu_array<T>::create(size_t size) 
{ cu_memory::create(size * elem_size); }
template<class T> inline void cu_array<T>::release()  
{ cu_memory::release(); }

template<class T> inline void cu_array<T>::copy_to(cu_array& other) const
{ cu_memory::copy_to(other); }
template<class T> inline void cu_array<T>::upload(const T *host_ptr, size_t size) 
{ cu_memory::upload(host_ptr, size * elem_size); }
template<class T> inline void cu_array<T>::download(T *host_ptr) const 
{ cu_memory::download( host_ptr ); }

template<class T> void cu_array<T>::swap(cu_array& other_arg) { cu_memory::swap(other_arg); }

template<class T> inline cu_array<T>::operator T*() { return ptr(); }
template<class T> inline cu_array<T>::operator const T*() const { return ptr(); }
template<class T> inline size_t cu_array<T>::size() const { return size_bytes() / elem_size; }

template<class T> inline       T* cu_array<T>::ptr()       { return cu_memory::ptr<T>(); }
template<class T> inline const T* cu_array<T>::ptr() const { return cu_memory::ptr<T>(); }

template<class T> template<class A> inline void cu_array<T>::upload(const std::vector<T, A>& data) { upload(&data[0], data.size()); }
template<class T> template<class A> inline void cu_array<T>::download(std::vector<T, A>& data) const { data.resize(size()); if (!data.empty()) download(&data[0]); }

///////////////////  Inline implementations of cu_array_host ////////////////////////////////////////////

template<class T> inline cu_array_host<T>::cu_array_host() {}
template<class T> inline cu_array_host<T>::cu_array_host(size_t size) : cu_memory_host(size * elem_size) {}
template<class T> inline cu_array_host<T>::cu_array_host(T *ptr, size_t size) : cu_memory_host(ptr, size * elem_size) {}
template<class T> inline cu_array_host<T>::cu_array_host(const cu_array_host& other) : cu_memory_host(other) {}
template<class T> inline cu_array_host<T>& cu_array_host<T>::operator=(const cu_array_host& other)
{ cu_memory_host::operator=(other); return *this; }

template<class T> inline void cu_array_host<T>::create(size_t size) 
{ cu_memory_host::create(size * elem_size); }
template<class T> inline void cu_array_host<T>::release()  
{ cu_memory_host::release(); }

template<class T> inline void cu_array_host<T>::copy_to(cu_array_host& other) const
{ cu_memory_host::copy_to(other); }
template<class T> inline void cu_array_host<T>::upload(const T *host_ptr, size_t size) 
{ cu_memory_host::upload(host_ptr, size * elem_size); }
template<class T> inline void cu_array_host<T>::download(T *host_ptr) const 
{ cu_memory_host::download( host_ptr ); }

template<class T> void cu_array_host<T>::swap(cu_array_host& other_arg) { cu_memory_host::swap(other_arg); }

template<class T> inline cu_array_host<T>::operator T*() { return ptr(); }
template<class T> inline cu_array_host<T>::operator const T*() const { return ptr(); }
template<class T> inline size_t cu_array_host<T>::size() const { return size_bytes() / elem_size; }

template<class T> inline       T* cu_array_host<T>::ptr()       { return cu_memory_host::ptr<T>(); }
template<class T> inline const T* cu_array_host<T>::ptr() const { return cu_memory_host::ptr<T>(); }

template<class T> template<class A> inline void cu_array_host<T>::upload(const std::vector<T, A>& data) { upload(&data[0], data.size()); }
template<class T> template<class A> inline void cu_array_host<T>::download(std::vector<T, A>& data) const { data.resize(size()); if (!data.empty()) download(&data[0]); }

/////////////////////  Inline implementations of cu_array_2d ////////////////////////////////////////////

template<class T> inline cu_array_2d<T>::cu_array_2d() {}
template<class T> inline cu_array_2d<T>::cu_array_2d(int rows, int cols) : cu_memory_2d(rows, cols * elem_size) {}
template<class T> inline cu_array_2d<T>::cu_array_2d(int rows, int cols, void *data, size_t step_bytes_) : cu_memory_2d(rows, cols * elem_size, data, step_bytes_) {}
template<class T> inline cu_array_2d<T>::cu_array_2d(const cu_array_2d& other) : cu_memory_2d(other) {}
template<class T> inline cu_array_2d<T>& cu_array_2d<T>::operator=(const cu_array_2d& other)
{ cu_memory_2d::operator=(other); return *this; }

template<class T> inline void cu_array_2d<T>::create(int rows, int cols) 
{ cu_memory_2d::create(rows, cols * elem_size); }
template<class T> inline void cu_array_2d<T>::release()  
{ cu_memory_2d::release(); }

template<class T> inline void cu_array_2d<T>::copy_to(cu_array_2d& other) const
{ cu_memory_2d::copy_to(other); }
template<class T> inline void cu_array_2d<T>::upload(const void *host_ptr, size_t host_step, int rows, int cols) 
{ cu_memory_2d::upload(host_ptr, host_step, rows, cols * elem_size); }
template<class T> inline void cu_array_2d<T>::download(void *host_ptr, size_t host_step) const 
{ cu_memory_2d::download( host_ptr, host_step ); }

template<class T> template<class A> inline void cu_array_2d<T>::upload(const std::vector<T, A>& data, int cols) 
{ upload(&data[0], cols * elem_size, data.size()/cols, cols); }

template<class T> template<class A> inline void cu_array_2d<T>::download(std::vector<T, A>& data, int& elem_step_) const 
{ elem_step_ = cols(); data.resize(cols() * rows()); if (!data.empty()) download(&data[0], cols_bytes());  }

template<class T> void  cu_array_2d<T>::swap(cu_array_2d& other_arg) { cu_memory_2d::swap(other_arg); }

template<class T> inline       T* cu_array_2d<T>::ptr(int y)       { return cu_memory_2d::ptr<T>(y); }
template<class T> inline const T* cu_array_2d<T>::ptr(int y) const { return cu_memory_2d::ptr<T>(y); }
            
template<class T> inline cu_array_2d<T>::operator T*() { return ptr(); }
template<class T> inline cu_array_2d<T>::operator const T*() const { return ptr(); }

template<class T> inline int cu_array_2d<T>::cols() const { return cu_memory_2d::cols_bytes()/elem_size; }
template<class T> inline int cu_array_2d<T>::rows() const { return cu_memory_2d::rows(); }

template<class T> inline size_t cu_array_2d<T>::elem_step() const { return cu_memory_2d::step()/elem_size; }

//#include <pcl/gpu/containers/initialization.h>
//#include <pcl/gpu/containers/device_array.h>
//using namespace pcl::device;

#endif /* !_CU_CONFIG_H */
