FROM python:3.10.12

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

RUN pip install -r requirements.txt

# Make port 4321 available to the world outside this container
EXPOSE 4334

CMD ["streamlit", "run", "--server.port", "4334", "presentation.py"]
