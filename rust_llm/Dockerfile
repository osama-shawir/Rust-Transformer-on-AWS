# Use an official Rust runtime as a parent image
FROM rust:latest as builder

# Set the working directory in the image to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Build the application
RUN cargo build --release

# Start a new stage with the same Rust image
FROM rust:latest

# Set the working directory in the image to /var/task
WORKDIR /var/task

# Copy the binary from the builder stage to the final stage
COPY --from=builder /app/target/release/rust_llm  ./bootstrap

# Copy the model file to the final stage
COPY --from=builder /app/src/pythia-410m-q5_1-ggjt.bin ./src/pythia-410m-q5_1-ggjt.bin

# The CMD instruction provides default execution parameters for an executing container
CMD ["./bootstrap"]