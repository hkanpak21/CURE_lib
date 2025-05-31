.PHONY: run

run:
	go run ./cmd/split_training_demo train

train-full-he:
	go run ./cmd/split_training_demo train --he --batches 10

eval:
	go run ./cmd/split_training_demo eval client_model.txt server_model.txt

clean:
	rm -f client_model.txt server_model.txt 