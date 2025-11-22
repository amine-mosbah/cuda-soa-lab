pipeline {
    agent any

    environment {
        IMAGE_NAME = "gpu-service-amine"
        STUDENT_PORT = "8125"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Test CUDA Kernel') {
            steps {
                sh 'python3 test_cuda.py'
            }
        }

        stage('Build Docker image') {
            steps {
                sh 'docker build -t ${IMAGE_NAME} .'
            }
        }

                stage('Deploy Container') {
            steps {
                sh '''
                    docker stop ${IMAGE_NAME} || true
                    docker rm ${IMAGE_NAME} || true

                    docker run -d --gpus all \
                        --name ${IMAGE_NAME} \
                        -e STUDENT_PORT=${STUDENT_PORT} \
                        -p ${STUDENT_PORT}:${STUDENT_PORT} \
                        -p 8000:8000 \
                        ${IMAGE_NAME}
                '''
            }
        }

        stage('Debug Container') {
            steps {
                sh '''
                    echo "==== docker ps ===="
                    docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Ports}}"

                    echo ""
                    echo "==== last 50 lines of gpu-service-amine logs ===="
                    docker logs --tail=50 gpu-service-amine || echo "No logs (container may have exited)"
                '''
            }
        }
    }
}   