
# Insurance prediction model

Insurance prediction model using Spark LinearRegression, designed for AWS EC2 instance deployment.

## Deployment

To deploy this project run

```bash
  docker-compose up --build
```

## API Reference

#### Get full JSON predictions

```http
  GET /
```

#### Get prediction given characteristics through JSON body

```http
  POST /predict
```



