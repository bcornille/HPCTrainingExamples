program openmp_rocblas
  use hipfort, only: hipsuccess, hipstreamcreate, hipstreamaddcallback
  use hipfort_hipblas, only: hipblascreate, hipblassetstream, hipblasdaxpy
  use hipfort_check, only: hipcheck, hipblascheck
  use iso_c_binding, only: c_ptr, c_funptr, c_loc, c_funloc, c_f_pointer
  use iso_fortran_env, only: int32, real64
  use omp_lib, only: omp_get_wtime, omp_event_handle_kind, omp_fulfill_event
  implicit none

  integer(int32) :: outer, inner, stream
  integer(int32) :: nouter = 4, ninner = 16*1024, nstreams = 4
  integer(omp_event_handle_kind), allocatable, target :: events(:)
  real(real64), PARAMETER :: a = 2.0
  type(c_ptr), allocatable :: streams(:), handles(:)
  real(real64), allocatable, target :: x(:,:), y(:,:)
  real(real64) :: timings(5)

  call read_input()
  allocate(streams(nstreams), handles(nouter), events(nouter))
  allocate(x(nouter, ninner), y(nouter, ninner))

  do stream = 0, nstreams-1
    call hipcheck(hipstreamcreate(streams(stream)))
  end do
  do outer = 1, nouter
    call hipblascheck(hipblascreate(handles(outer)))
  end do

  !$omp target enter data map(alloc:x,y)

  timings(1) = omp_get_wtime()

  write(*,*) "Start first loop"
  do outer = 1, nouter
    stream = mod(outer,nstreams)
    !$omp target teams distribute parallel do nowait depend(out:streams(stream))
    do inner = 1, ninner
      x(outer,inner) = 0.125_real64
      y(outer,inner) = 0.25_real64
    end do
  end do
  write(*,*) "End first loop"

  timings(2) = omp_get_wtime()

  write(*,*) "Start second loop"
  do outer = 1, nouter
    if (nstreams > 1) then
      stream = mod(outer,nstreams) + 1
    else
      stream = 1
    end if
    block
      interface
        subroutine callback(stream, status, cb_dat) bind(c)
          use iso_c_binding, only: c_ptr
          type(c_ptr), value :: stream, cb_dat
          integer(kind(hipsuccess)), value :: status
        end subroutine callback
      end interface
      integer(omp_event_handle_kind) :: event
      write(*,*) "Before task: ", event
      write(*,*) "  location: ", loc(event)
      !$omp task depend(inout:streams(stream)) detach(event)
      call hipblascheck(hipblassetstream(handles(outer), streams(stream)))
      !$omp target data use_device_addr(x,y)
      call hipblascheck(hipblasdaxpy(handles(outer), ninner, a, c_loc(x(outer,1)), 1, c_loc(y(outer,1)), 1))
      !$omp end target data
      write(*,*) "In task: ", event
      write(*,*) "  location: ", loc(event)
      call hipcheck(hipstreamaddcallback(streams(stream), c_funloc(callback), c_loc(event), 0))
      !$omp end task
      events(nouter) = event
      write(*,*) "After task: ", event
      write(*,*) "  location: ", loc(event)
    end block
  end do
  write(*,*) "End second loop"

  timings(3) = omp_get_wtime()

  write(*,*) "Start third loop"
  do outer = 1, nouter
    stream = mod(outer,nstreams)
    !$omp target teams distribute parallel do nowait depend(in:streams(stream))
    do inner = 1, ninner
      y(outer, inner) = y(outer, inner) - a*x(outer, inner)
    end do
  end do
  write(*,*) "End third loop"

  timings(4) = omp_get_wtime()

  write(*,*) "Waiting"
  !$omp taskwait

  timings(5) = omp_get_wtime()

  !$omp target exit data map(delete:x,y)

  write(*,*) "First loop submit time: ", timings(2) - timings(1)
  write(*,*) "Second loop submit time: ", timings(3) - timings(2)
  write(*,*) "Third loop submit time: ", timings(4) - timings(3)
  write(*,*) "Wait time: ", timings(5) - timings(4)
  write(*,*) "Total time: ", timings(5) - timings(1)

contains

  subroutine read_input()
    integer(int32) :: nargs, iarg
    character(len=64) :: option, value
    nargs = command_argument_count()
    do iarg = 1, nargs, 2
      call get_command_argument(iarg,option)
      select case(option)
      case("-n","--num-regions")
        call get_command_argument(iarg+1, value)
        read(value,'(i8)') nouter
      case("-s","--num-streams")
        call get_command_argument(iarg+1, value)
        read(value,'(i8)') nstreams
      case("-w","--work-size")
        call get_command_argument(iarg+1, value)
        read(value,'(i8)') ninner
      case default
        call print_usage()
        error stop "Unrecognized command "//trim(option)
      end select
    end do
  end subroutine read_input

  subroutine print_usage()
    write(*,*) "Usage: detach_example <args>"
    write(*,*) "  -n,--num-regions    number of outer loop trips"
    write(*,*) "  -w,--work-size      number of inner loop trips"
    write(*,*) "  -s,--num-streams    number of streams to utilize"
  end subroutine

  subroutine callback(stream, status, cb_dat) bind(c)
    type(c_ptr), value :: stream, cb_dat
    integer(kind(hipsuccess)), value :: status
    integer(omp_event_handle_kind), pointer :: event
    call c_f_pointer(cb_dat, event)
    write(*,*) "Staus: ", status, "/", hipsuccess
    write(*,*) "In callback: ", event
      write(*,*) "  location: ", loc(event)
    call omp_fulfill_event(event)
  end subroutine callback

end program openmp_rocblas
