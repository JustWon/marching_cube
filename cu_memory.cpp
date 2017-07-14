/**
	\file		cu_memory.cpp
	\brief		Memory functions
	\author		Seongoh Lee
	\version	1.0
	\date		2011.12.09
*/

// includes, project
#include "cu_memory.hpp"

//////////////////////////    XADD    ///////////////////////////////

#ifdef __GNUC__
    
    #if __GNUC__*10 + __GNUC_MINOR__ >= 42

        #if !defined WIN32 && (defined __i486__ || defined __i586__ || defined __i686__ || defined __MMX__ || defined __SSE__  || defined __ppc__)
            #define CV_XADD __sync_fetch_and_add
        #else
            #include <ext/atomicity.h>
            #define CV_XADD __gnu_cxx::__exchange_and_add
        #endif
    #else
        #include <bits/atomicity.h>
        #if __GNUC__*10 + __GNUC_MINOR__ >= 34
            #define CV_XADD __gnu_cxx::__exchange_and_add
        #else
            #define CV_XADD __exchange_and_add
        #endif
  #endif
    
#elif defined WIN32 || defined _WIN32
    #include <intrin.h>
    #define CV_XADD(addr,delta) _InterlockedExchangeAdd((long volatile*)(addr), (delta))
#else

    template<typename _Tp> static inline _Tp CV_XADD(_Tp* addr, _Tp delta)
    { int tmp = *addr; *addr += delta; return tmp; }
    
#endif

////////////////////////    DeviceArray    /////////////////////////////
    
cu_memory::cu_memory() : data_(0), size_bytes_(0), refcount_(0) {}
cu_memory::cu_memory(void *ptr_arg, size_t size_bytes_arg) : data_(ptr_arg), size_bytes_(size_bytes_arg), refcount_(0){}
cu_memory::cu_memory(size_t size_bytes_arg)  : data_(0), size_bytes_(0), refcount_(0) { create(size_bytes_arg); }
cu_memory::~cu_memory() { release(); }

cu_memory::cu_memory(const cu_memory& other_arg) 
    : data_(other_arg.data_), size_bytes_(other_arg.size_bytes_), refcount_(other_arg.refcount_)
{
    if( refcount_ )
        CV_XADD(refcount_, 1);
}

cu_memory& cu_memory::operator = (const cu_memory& other_arg)
{
    if( this != &other_arg )
    {
        if( other_arg.refcount_ )
            CV_XADD(other_arg.refcount_, 1);
        release();
        
        data_      = other_arg.data_;
        size_bytes_ = other_arg.size_bytes_;                
        refcount_  = other_arg.refcount_;
    }
    return *this;
}

void cu_memory::create(size_t size_bytes_arg)
{
    if (size_bytes_arg == size_bytes_)
        return;
            
    if( size_bytes_arg > 0)
    {        
        if( data_ )
            release();

        size_bytes_ = size_bytes_arg;
                        
        checkCudaErrors( cudaMalloc(&data_, size_bytes_) );        

        //refcount_ = (int*)cv::fastMalloc(sizeof(*refcount_));
        refcount_ = new int;
        *refcount_ = 1;
    }
}

void cu_memory::copy_to(cu_memory& other) const
{
    if (empty())
        other.release();
    else
    {    
        other.create(size_bytes_);    
        checkCudaErrors( cudaMemcpy(other.data_, data_, size_bytes_, cudaMemcpyDeviceToDevice) );
        checkCudaErrors( cudaDeviceSynchronize() );
    }
}

void cu_memory::release()
{
    if( refcount_ && CV_XADD(refcount_, -1) == 1 )
    {
        //cv::fastFree(refcount);
        delete refcount_;
        checkCudaErrors( cudaFree(data_) );
    }
    data_ = 0;
    size_bytes_ = 0;
    refcount_ = 0;
}

void cu_memory::upload(const void *host_ptr_arg, size_t size_bytes_arg)
{
    create(size_bytes_arg);
    checkCudaErrors( cudaMemcpy(data_, host_ptr_arg, size_bytes_, cudaMemcpyHostToDevice) );        
}

void cu_memory::download(void *host_ptr_arg) const
{    
    checkCudaErrors( cudaMemcpy(host_ptr_arg, data_, size_bytes_, cudaMemcpyDeviceToHost) );
}          

void cu_memory::swap(cu_memory& other_arg)
{
    std::swap(data_, other_arg.data_);
    std::swap(size_bytes_, other_arg.size_bytes_);
    std::swap(refcount_, other_arg.refcount_);
}

bool cu_memory::empty() const { return !data_; }
size_t cu_memory::size_bytes() const { return size_bytes_; }

////////////////////////    HostArray    /////////////////////////////
    
cu_memory_host::cu_memory_host() : data_(0), size_bytes_(0), refcount_(0) {}
cu_memory_host::cu_memory_host(void *ptr_arg, size_t size_bytes_arg) : data_(ptr_arg), size_bytes_(size_bytes_arg), refcount_(0){}
cu_memory_host::cu_memory_host(size_t size_bytes_arg)  : data_(0), size_bytes_(0), refcount_(0) { create(size_bytes_arg); }
cu_memory_host::~cu_memory_host() { release(); }

cu_memory_host::cu_memory_host(const cu_memory_host& other_arg) 
    : data_(other_arg.data_), size_bytes_(other_arg.size_bytes_), refcount_(other_arg.refcount_)
{
    if( refcount_ )
        CV_XADD(refcount_, 1);
}

cu_memory_host& cu_memory_host::operator = (const cu_memory_host& other_arg)
{
    if( this != &other_arg )
    {
        if( other_arg.refcount_ )
            CV_XADD(other_arg.refcount_, 1);
        release();
        
        data_      = other_arg.data_;
        size_bytes_ = other_arg.size_bytes_;                
        refcount_  = other_arg.refcount_;
    }
    return *this;
}

void cu_memory_host::create(size_t size_bytes_arg)
{
    if (size_bytes_arg == size_bytes_)
        return;
            
    if( size_bytes_arg > 0)
    {        
        if( data_ )
            release();

        size_bytes_ = size_bytes_arg;
                        
        checkCudaErrors( cudaMallocHost(&data_, size_bytes_) );        

        //refcount_ = (int*)cv::fastMalloc(sizeof(*refcount_));
        refcount_ = new int;
        *refcount_ = 1;
    }
}

void cu_memory_host::copy_to(cu_memory_host& other) const
{
    if (empty())
        other.release();
    else
    {    
        other.create(size_bytes_);    
        checkCudaErrors( cudaMemcpy(other.data_, data_, size_bytes_, cudaMemcpyHostToHost) );
        checkCudaErrors( cudaDeviceSynchronize() );
    }
}

void cu_memory_host::release()
{
    if( refcount_ && CV_XADD(refcount_, -1) == 1 )
    {
        //cv::fastFree(refcount);
        delete refcount_;
        checkCudaErrors( cudaFreeHost(data_) );
    }
    data_ = 0;
    size_bytes_ = 0;
    refcount_ = 0;
}

void cu_memory_host::upload(const void *device_ptr_arg, size_t size_bytes_arg)
{
    create(size_bytes_arg);
    checkCudaErrors( cudaMemcpy(data_, device_ptr_arg, size_bytes_, cudaMemcpyDeviceToHost) );        
}

void cu_memory_host::download(void *device_ptr_arg) const
{    
    checkCudaErrors( cudaMemcpy(device_ptr_arg, data_, size_bytes_, cudaMemcpyHostToDevice) );
}

void cu_memory_host::swap(cu_memory_host& other_arg)
{
    std::swap(data_, other_arg.data_);
    std::swap(size_bytes_, other_arg.size_bytes_);
    std::swap(refcount_, other_arg.refcount_);
}

bool cu_memory_host::empty() const { return !data_; }
size_t cu_memory_host::size_bytes() const { return size_bytes_; }

////////////////////////    DeviceArray2D    /////////////////////////////

cu_memory_2d::cu_memory_2d() : data_(0), step_(0), cols_bytes_(0), rows_(0), refcount_(0) {}

cu_memory_2d::cu_memory_2d(int rows_arg, int cols_bytes_arg) 
    : data_(0), step_(0), cols_bytes_(0), rows_(0), refcount_(0)
{ 
    create(rows_arg, cols_bytes_arg); 
}

cu_memory_2d::cu_memory_2d(int rows_arg, int cols_bytes_arg, void *data_arg, size_t step_arg) 
    :  data_(data_arg), step_(step_arg), cols_bytes_(cols_bytes_arg), rows_(rows_arg), refcount_(0) {}

cu_memory_2d::~cu_memory_2d() { release(); }

cu_memory_2d::cu_memory_2d(const cu_memory_2d& other_arg) : 
    data_(other_arg.data_), step_(other_arg.step_), cols_bytes_(other_arg.cols_bytes_), rows_(other_arg.rows_), refcount_(other_arg.refcount_)
{
    if( refcount_ )
        CV_XADD(refcount_, 1);
}

cu_memory_2d& cu_memory_2d::operator = (const cu_memory_2d& other_arg)
{
    if( this != &other_arg ) {
        if( other_arg.refcount_ )
            CV_XADD(other_arg.refcount_, 1);
        release();
        
        cols_bytes_ = other_arg.cols_bytes_;
        rows_ = other_arg.rows_;
        data_ = other_arg.data_;
        step_ = other_arg.step_;
                
        refcount_ = other_arg.refcount_;
    }
    return *this;
}

void cu_memory_2d::create(int rows_arg, int cols_bytes_arg)
{
    if (cols_bytes_ == cols_bytes_arg && rows_ == rows_arg)
        return;
            
    if( rows_arg > 0 && cols_bytes_arg > 0) {        
        if( data_ )
            release();
              
        cols_bytes_ = cols_bytes_arg;
        rows_ = rows_arg;
          
		//data_ = malloc( cols_bytes_ * rows_ ); step_ = cols_bytes_; // FIXME
		checkCudaErrors( cudaMallocPitch( (void**)&data_, &step_, cols_bytes_, rows_) );        

        //refcount = (int*)cv::fastMalloc(sizeof(*refcount));
        refcount_ = new int;
        *refcount_ = 1;
    }
}

void cu_memory_2d::release()
{
    if( refcount_ && CV_XADD(refcount_, -1) == 1 ) {
        //cv::fastFree(refcount);
        delete refcount_;
		//free( data_ ); // FIXME
        checkCudaErrors( cudaFree(data_) );
    }

    cols_bytes_ = 0;
    rows_ = 0;    
    data_ = 0;    
    step_ = 0;
    refcount_ = 0;
}

void cu_memory_2d::copy_to(cu_memory_2d& other) const
{
    if (empty())
        other.release();
    else {
        other.create(rows_, cols_bytes_);
		//memcpy( other.data_, data_, cols_bytes_ * rows_ );
        checkCudaErrors( cudaMemcpy2D(other.data_, other.step_, data_, step_, cols_bytes_, rows_, cudaMemcpyDeviceToDevice) );
        checkCudaErrors( cudaDeviceSynchronize() );
    }
}

void cu_memory_2d::upload(const void *host_ptr_arg, size_t host_step_arg, int rows_arg, int cols_bytes_arg)
{
    create(rows_arg, cols_bytes_arg);
    checkCudaErrors( cudaMemcpy2D(data_, step_, host_ptr_arg, host_step_arg, cols_bytes_, rows_, cudaMemcpyHostToDevice) );        
}

void cu_memory_2d::download(void *host_ptr_arg, size_t host_step_arg) const
{    
    checkCudaErrors( cudaMemcpy2D(host_ptr_arg, host_step_arg, data_, step_, cols_bytes_, rows_, cudaMemcpyDeviceToHost) );
}      

void cu_memory_2d::swap(cu_memory_2d& other_arg)
{    
    std::swap(data_, other_arg.data_);
    std::swap(step_, other_arg.step_);

    std::swap(cols_bytes_, other_arg.cols_bytes_);
    std::swap(rows_, other_arg.rows_);
    std::swap(refcount_, other_arg.refcount_);                 
}

bool cu_memory_2d::empty() const { return !data_; }
int cu_memory_2d::cols_bytes() const { return cols_bytes_; }
int cu_memory_2d::rows() const { return rows_; }
size_t cu_memory_2d::step() const { return step_; }
