pipeline {
    agent any
    environment {
        APP_NAME = "FeedbackManagementSystem"
    }
    stages {
        stage('Clone Repository') {
            steps {
                echo "Cloning repository..."
                git url: 'https://github.com/sathwikshetty33/FeedbackManagementSystem', branch: 'main'
            }
        }
        
        stage('Docker-compose-down') {
            steps {
                sh 'cd core && docker-compose down --remove-orphans'
                echo "Taking down all containers and removing orphans"
            }
        }   
        stage('Docker-compose-up') {
            steps {
                echo "Creating docker containers"
                sh '''
                    cd core
                    docker-compose up -d --build
                '''
            }
        }
    }
    
    post {
        success {
            echo "Pipeline executed successfully!"
        }
        failure {
            echo "Pipeline failed!"
        }
    }
}