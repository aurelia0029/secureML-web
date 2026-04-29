FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    cmake \
    git \
    wget \
    python3 \
    python3-pip \
    apache2 \
    apache2-dev \
    libssl-dev \
    pkg-config \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build and install liboqs-custom (with hardware acceleration)
WORKDIR /tmp
COPY liboqs-custom /tmp/liboqs-custom
RUN cd liboqs-custom && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Build and install oqs-provider
RUN git clone --depth 1 https://github.com/open-quantum-safe/oqs-provider.git && \
    cd oqs-provider && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/usr/local && \
    cmake --build build && \
    cmake --install build && \
    cd /tmp && rm -rf oqs-provider

# Configure OpenSSL to load OQS provider
RUN mkdir -p /usr/local/ssl
COPY openssl-oqs.cnf /usr/local/ssl/openssl.cnf

# Set OpenSSL config environment
ENV OPENSSL_CONF=/usr/local/ssl/openssl.cnf
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Set Apache environment variables
ENV APACHE_RUN_DIR=/var/run/apache2
ENV APACHE_LOCK_DIR=/var/lock/apache2
ENV APACHE_LOG_DIR=/var/log/apache2
RUN mkdir -p $APACHE_RUN_DIR $APACHE_LOCK_DIR $APACHE_LOG_DIR

# Enable Apache modules
RUN a2enmod ssl proxy proxy_http headers rewrite

# Copy Apache configuration
COPY apache-pq.conf /etc/apache2/sites-available/000-default.conf

# Copy certificates
RUN mkdir -p /etc/apache2/certs
COPY certs/server_cert.pem /etc/apache2/certs/
COPY certs/server_key.pem /etc/apache2/certs/

# Expose HTTPS port
EXPOSE 8443

# Start Apache in foreground
CMD ["apache2ctl", "-D", "FOREGROUND"]
