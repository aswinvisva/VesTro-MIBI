read -p "Enter the Data Directory: " dir

docker build -t vestro-mibi .

docker run -v "$dir:/data_dir" -p 8888:8888 vestro-mibi