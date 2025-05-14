import os
import shutil
import requests
from git import Repo
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

GITLAB_TOKEN = os.getenv("GITLAB_TOKEN")
GROUP_URL = os.getenv("GITLAB_GROUP_URL")
GROUP_HOST = GROUP_URL.split("//")[1].split("/")[0]
GROUP_FULL_PATH = "/".join(GROUP_URL.split("//")[1].split("/")[1:])
CLONE_BASE_DIR = "repos"

VALID_EXTENSIONS = (
    ".md", ".py", ".env", ".yml", ".yaml", ".txt", ".json",
    ".js", ".cs"
)
SPECIAL_FILES = ("Dockerfile", ".dockerignore")

HEADERS = {"PRIVATE-TOKEN": GITLAB_TOKEN}


def get_group_id(full_path):
    url = f"https://{GROUP_HOST}/api/v4/groups/{full_path}"
    res = requests.get(url, headers=HEADERS)
    res.raise_for_status()
    return res.json()["id"]


def get_all_projects(group_id):
    all_projects = []

    def recurse(gid):
        proj_url = f"https://{GROUP_HOST}/api/v4/groups/{gid}/projects?per_page=100&include_subgroups=true"
        res = requests.get(proj_url, headers=HEADERS)
        res.raise_for_status()
        all_projects.extend(res.json())

        subgroup_url = f"https://{GROUP_HOST}/api/v4/groups/{gid}/subgroups?per_page=100"
        res = requests.get(subgroup_url, headers=HEADERS)
        res.raise_for_status()
        for subgroup in res.json():
            recurse(subgroup["id"])

    recurse(group_id)
    return all_projects


def clone_repos():
    group_id = get_group_id(GROUP_FULL_PATH)
    projects = get_all_projects(group_id)

    for project in tqdm(projects, desc="Clonando repositórios"):
        repo_url = project["ssh_url_to_repo"]
        relative_path = project["path_with_namespace"]
        dest = os.path.join(CLONE_BASE_DIR, relative_path)

        if os.path.exists(dest):
            print(f"[SKIP] Já existe: {relative_path}")
            continue

        os.makedirs(os.path.dirname(dest), exist_ok=True)

        try:
            Repo.clone_from(repo_url, dest)
        except Exception as e:
            print(f"[ERRO] Falha ao clonar {repo_url}: {e}")


def ingest_to_chroma():
    documents = []
    for root, _, files in os.walk(CLONE_BASE_DIR):
        for file in files:
            if file.endswith(VALID_EXTENSIONS) or file in SPECIAL_FILES:
                path = os.path.join(root, file)
                try:
                    loader = TextLoader(path, encoding="utf-8")
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"[ERRO] Falha ao carregar {path}: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    print(f"[INFO] Ingerindo {len(chunks)} chunks no ChromaDB...")
    Chroma.from_documents(
        chunks,
        OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory="./chroma_db"
    )


if __name__ == "__main__":
    clone_repos()
    ingest_to_chroma()
