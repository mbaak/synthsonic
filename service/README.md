# Synthsonic UI - Synthetic data as a service (SDaaS)

## Components

### UI

#### Running UI

##### Localhost

```shell script
make run-ui-local
```

##### Docker

```shell script
docker-compose build --force ui
docker-compose up -d ui
```

### API

REST API is written in FastAPI and provides simple pipeline creating model and generating synthetic data.

#### Configuration options

API is configurable through env variables and enabled following configuration options:
* `DISPLAY_ROWS` - How many rows of data to show in UI. Default = 5.
* `INPUT_DATA_MIN_ROWS` - Minimum number of rows in input data. Default = 1000.

#### Running API

##### Localhost

```shell script
make run-api-local
```

##### Docker

```shell script
docker-compose build --force api
docker-compose up -d api
```