# Jehyeon's Profile page

This is Jehyeon's personal blog for coding records.

---

## Getting Started & Prerequisites

### 1. 요구사항 (Prerequisites)
* **Hugo** (v0.169.0)
* **Git** (v2.54.0)
* **Go** (v.1.27.3)

### 2. 설치 및 실행 절차 (Installation & Local Run)
저장소를 클론한 후 로컬 환경에서 사이트를 빌드하고 미리 볼 수 있는 명령어입니다.

#### I. 설치 절차 (Install)
* **Hugo (Extended 버전) 설치 및 버전 확인**
  * **설치 방법:** 
    * macOS: `brew install hugo`
    * Windows: `choco install hugo-extended`
    * Linux: `sudo apt-get install hugo`
  * *설치 확인:* 터미널에 `hugo version`을 입력하여 `extended` 버전이 정상 설치되었는지 확인합니다.

<br>

* **Git 설치 및 버전 확인**
  * **설치 방법:** [Git 공식 홈페이지](https://git-scm.com/) 또는 패키지 매니저(`brew install git`, `sudo apt-get install git`)를 통해 설치합니다.

<br>

* **Go (Golang) 설치 및 버전 확인** *(일부 Hugo 테마 모듈 및 확장 기능 사용 시 필요)*
  * **설치 방법:** [Go 공식 홈페이지](https://golang.org/)에서 다운로드하거나 패키지 매니저(`brew install go`, `sudo apt-get install golang`)를 통해 설치합니다.

<br>

#### II. 프로젝트 클론 및 실행 경로 (Build & Run)
준비된 환경에서 아래 명령어를 순서대로 입력하여 로컬 서버를 실행합니다.

```bash
# 1. 저장소 클론 (서브모듈 포함)
git clone --recurse-submodules https://github.com/yoosi13/MyProfile.git
cd MyProfile

# 2. 로컬 개발 서버 실행
hugo -s site server --baseURL "http://localhost:1313/" --port 1313
(만약 1313포트가 이미 사용중이면 터미널에 출력된 Local 주소로 접속해주세요.)
```

## 3. Configuration & Deployment

### 1. 수정할 파일 및 설정
* 수정할 파일 위치: /site/config/_default/hugo.toml
* baseURL을 배포환경(Github Pages)에 맞는 URL로 입력해주세요.
```toml
형식: baseURL = "https://[사용자명].github.io/[레포지토리명]/"
baseURL = "https://yoosi13.github.io/MyProfile/"
```

### 2. 수정 내용 커밋 및 푸시 (Commit & Push)
```bash
git add .
git commit -m "docs: update site configuration and content"
git push origin main
```

### 3. 배포 결과 및 공개 화면 확인 (Actions & Public site)
* Github Settings -> Pages에서 Build and deployment의 source를 GitHub Actions로 설정
* Actions 탭에서 pages-build-deployment 워크플로우가 초록색 체크로 성공했는지 확인

* 이후 성공한 워크플로우를 열어 홈페이지를 확인하면 됩니다.

## License & Attribution

### 1. 사용 테마 및 라이선스 고지 (Theme & License)
* **사용 테마:** [Hugo Theme Stack](https://github.com/CaiJimmy/hugo-theme-stack) (Designed by [Jimmy](https://github.com/CaiJimmy))
* **라이선스:** **GNU General Public License v3.0 (GPL-3.0)**

### 2. 콘텐츠 및 작성 출처 구분 (Author & AI Attribution)
* **직접 작성/제작 자료:**
  * 개인 프로필 정보 및 포트폴리오 이력 내용
  * 커스텀 Hugo 설정(`hugo.toml`) 및 블로그 게시글 본문
* **AI 생성 및 보조 자료:**
  * README.md 내 재현·배포 실행 가이드 작성 및 명령어 가공 보조, 원본 페이지 영번역 (Gemini 활용)

---

# Hugo Theme Stack

![image](https://user-images.githubusercontent.com/5889006/190859441-141b5f81-8483-40d2-bd96-ebf85616a46d.png)

Card-style Hugo theme designed for bloggers.

## Quickstart

Use this template: [CaiJimmy/hugo-theme-stack-starter](https://github.com/CaiJimmy/hugo-theme-stack-starter)

## Demo

Check [demo.stack.cai.im](https://demo.stack.cai.im)

## Documentation

- [English](https://stack.cai.im)
- [中文](https://stack.cai.im/zh)

## Copyright

**Licensed under the GNU General Public License v3.0**

Please do not remove the "*Theme Stack designed by Jimmy*" text and link.

If you want to port this theme to another blogging platform, please let me know🙏.
