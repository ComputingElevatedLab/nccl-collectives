.PHONY : all

all:
	$(MAKE) -C mpi-ata
	$(MAKE) -C mpi-ata-bruck
	$(MAKE) -C mpi-ata-spreadout
	$(MAKE) -C nccl-ata
	$(MAKE) -C nccl-ata-bruck
	$(MAKE) -C mpi-ata-spreadout

clean:
	$(MAKE) -C mpi-ata clean
	$(MAKE) -C mpi-ata-bruck clean
	$(MAKE) -C mpi-ata-spreadout clean
	$(MAKE) -C nccl-ata clean
	$(MAKE) -C nccl-ata-bruck clean
	$(MAKE) -C mpi-ata-spreadout clean