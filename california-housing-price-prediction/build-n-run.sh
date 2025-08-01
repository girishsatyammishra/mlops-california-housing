# Build Docker Image
docker build -t california-housing-price-prediction .
# Run
docker run -d -p 5001:5001 california-housing-price-prediction
