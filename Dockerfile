# 1. Base Image 선택
# 파이썬 3.12 버전의 경량화된 리눅스(slim) 이미지를 베이스로 사용하여 이미지 용량을 최적화
FROM python:3.12-slim

# 2. 작업 디렉토리(Work Directory) 설정
# 컨테이너 내부에서 명령어가 실행될 기본 폴더 위치를 /app으로 지정
WORKDIR /app

# 3. 시스템 의존성 패키지 설치
# 빌드에 필요한 최소한의 시스템 패키지를 설치.
# apt-get update 후 설치가 완료되면 캐시 패키지 파일(lists)을 삭제하여 이미지 용량을 줄임
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. 의존성 라이브러리 파일 복사 및 설치 (Docker 캐시 레이어 활용)
# CPU 전용 PyTorch를 명시적으로 먼저 설치하여 수 GB에 달하는 불필요한 CUDA 라이브러리 다운로드를 차단
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

# --no-cache-dir 옵션을 사용하여 pip 캐시를 저장하지 않고 설치하여 컨테이너 이미지를 한 번 더 경량화
RUN pip install --no-cache-dir -r requirements.txt

# 5. 애플리케이션 전체 소스 코드 복사
# 현재 로컬 디렉토리의 모든 파일(.env 제외, .dockerignore 처리 권장)을 컨테이너 내부 /app 폴더로 복사
COPY . .

# 6. 외부 노출 포트 명시
# 컨테이너가 런타임에 8000번 포트를 사용함을 문서화 용도로 명시 (FastAPI 기본 포트)
EXPOSE 8000

# 7. 컨테이너 실행 기본 명령어 (Entrypoint Command)
# 컨테이너가 시작될 때 Uvicorn ASGI 서버를 구동
# --host 0.0.0.0: 컨테이너 외부(EC2, Spring Boot 등)에서 접속할 수 있도록 모든 IP 대역의 바인딩을 허용
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]