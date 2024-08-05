program openmp_rocblas
  use hipfort, only: hipsuccess, hipstreamcreate, hipdevicesynchronize, hipstreamaddcallback, hipstreamdestroy
  use hipfort_hipblas, only: hipblascreate, hipblassetstream, hipblasdaxpy, hipblasdestroy
  use hipfort_check, only: hipcheck, hipblascheck
  use iso_c_binding, only: c_ptr, c_funptr, c_loc, c_funloc, c_f_pointer
  use iso_fortran_env, only: int32, real64
  use omp_lib, only: omp_get_wtime, omp_event_handle_kind, omp_fulfill_event
  implicit none

  integer(int32) :: inner
  integer(int32) :: ninner = 16*1024
  integer(omp_event_handle_kind) :: event
  real(real64), PARAMETER :: a = 2.0
  type(c_ptr) :: streams(2), hbhandle
  real(real64), allocatable, target :: x(:,:), y(:,:)
  real(real64) :: timings(6)

  call read_input()
  write(*,*) "Work size: ", ninner
  allocate(x(ninner, 2), y(ninner, 2))

  call hipcheck(hipstreamcreate(streams(1)))
  call hipcheck(hipstreamcreate(streams(2)))
  call hipblascheck(hipblascreate(hbhandle))
  call hipblascheck(hipblassetstream(hbhandle, streams(2)))

  !write(*,*) "Allocate device memory."
  !$omp target enter data map(alloc:x,y)

  !write(*,*) "Warmup OpenMP kernel."
  !$omp target teams distribute parallel do
  do inner = 1, ninner
    x(inner, 2) = 0.125_real64
    y(inner, 2) = 0.25_real64
  enddo

  !write(*,*) "Warmup hipBLAS call."
  !$omp target data use_device_addr(x,y)
  call hipblascheck(hipblasdaxpy(hbhandle, ninner, a, c_loc(x(1,1)), 1, c_loc(y(1,1)), 1))
  !$omp end target data
  call hipcheck(hipdevicesynchronize())
  

  timings(1) = omp_get_wtime()

  !write(*,*) "Start first kernel"
  !$omp target teams distribute parallel do nowait depend(out:streams(1))
  !!$omp target teams distribute parallel do nowait depend(out:x(:,1),y(:,1))
  do inner = 1, ninner
    x(inner, 1) = 0.125_real64
    y(inner, 1) = 0.25_real64
  end do
  !write(*,*) "End first kernel"

  timings(2) = omp_get_wtime()

  !write(*,*) "Start second kernel"
  write(*,*) "Before task: ", event
  write(*,*) "  location: ", loc(event)
  !$omp task depend(inout:streams(2)) detach(event)
  !!$omp task depend(inout:x(:,2),y(:,2)) detach(event)
  !$omp target data use_device_addr(x,y)
  call hipblascheck(hipblasdaxpy(hbhandle, ninner, a, c_loc(x(1, 2)), 1, c_loc(y(1, 2)), 1))
  !$omp end target data
  write(*,*) "In task: ", event
  write(*,*) "  location: ", loc(event)
  call hipcheck(hipstreamaddcallback(streams(2), c_funloc(callback), c_loc(event), 0))
  !$omp end task
  write(*,*) "After task: ", event
  write(*,*) "  location: ", loc(event)
  !write(*,*) "End second kernel"
  !write(*,*) "Events :", events

  timings(3) = omp_get_wtime()

  !$omp taskwait depend(inout:streams(1))

  timings(4) = omp_get_wtime()

  !write(*,*) "Start third kernel"
  !$omp target teams distribute parallel do nowait depend(in:streams(1))
  !!$omp target teams distribute parallel do nowait depend(in:x(:,1)) depend(inout:y(:,1))
  do inner = 1, ninner
    y(inner, 1) = y(inner, 1) + a*x(inner, 1)
  end do
  !write(*,*) "End third loop"

  timings(5) = omp_get_wtime()

  !write(*,*) "Waiting"
  !$omp taskwait

  timings(6) = omp_get_wtime()

  !$omp target update from(y)
  write(*,*) y

  !$omp target exit data map(delete:x,y)
  call hipblascheck(hipblasdestroy(hbhandle))
  call hipcheck(hipstreamdestroy(streams(2)))
  call hipcheck(hipstreamdestroy(streams(1)))

  write(*,*) "First kernel submit time: ", timings(2) - timings(1)
  write(*,*) "Second kernel submit time: ", timings(3) - timings(2)
  write(*,*) "First kernel wait time: ", timings(4) - timings(3)
  write(*,*) "Third kernel submit time: ", timings(5) - timings(4)
  write(*,*) "Wait time: ", timings(6) - timings(5)
  write(*,*) "Total time: ", timings(6) - timings(1)

contains

  subroutine read_input()
    integer(int32) :: nargs, iarg
    character(len=64) :: option, value
    nargs = command_argument_count()
    do iarg = 1, nargs, 2
      call get_command_argument(iarg,option)
      select case(option)
      case("-w","--work-size")
        call get_command_argument(iarg+1, value)
        read(value,'(i64)') ninner
      case default
        call print_usage()
        error stop "Unrecognized command "//trim(option)
      end select
    end do
  end subroutine read_input

  subroutine print_usage()
    write(*,*) "Usage: detach_example <args>"
    write(*,*) "  -w,--work-size      number of inner loop trips"
  end subroutine

  subroutine callback(stream, status, cb_dat) bind(c)
    type(c_ptr), value :: stream, cb_dat
    integer(kind(hipsuccess)), value :: status
    integer(omp_event_handle_kind), pointer :: event
    call c_f_pointer(cb_dat, event)
    !write(*,*) "Staus: ", status, "/", hipsuccess
    !write(*,*) "In callback: ", event
    !write(*,*) "  location: ", loc(event)
    call omp_fulfill_event(event)
  end subroutine callback

end program openmp_rocblas
