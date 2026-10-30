# Publish this GitHub repository

Create a **public** repository named `Evidence-Graph-Reranker` under `AngshulMajumdar` (or use another name you prefer). Upload the **contents** of the extracted ZIP to the repository root, preserving all nested folders. Upload using GitHub's web editor or Git; do not upload the ZIP as one GitHub file.

Your repository needs no GitHub Actions, hidden files, hidden directories, build scripts, Jekyll plugins, or secret tokens. It contains a standalone `docs/index.html` and `docs/styles.css`.

## Enable GitHub Pages

1. Open your new GitHub repository → **Settings** → **Pages**.
2. Under **Build and deployment**, choose **Deploy from a branch**.
3. Choose branch **main**, folder **/docs**, then click **Save**.
4. Once published, the public page will be at `https://AngshulMajumdar.github.io/Evidence-Graph-Reranker/` if the repository name is exactly `Evidence-Graph-Reranker`.

The GitHub README contains numerical tables rather than the long-form figures; the article and all six PNG files are under `docs/`.

## Release tagging

After testing, create a GitHub Release (Releases → Draft a new release), tag `v0.1.0`, title `Evidence Graph Reranker v0.1.0`, and describe the single fixed selector, CPU + offline interfaces, Python/CLI/Gradio APIs, and six-benchmark evaluation. You can attach this ZIP as a release asset.

## Hugging Face Space

Keep the existing Space at `https://huggingface.co/spaces/Angshul/Evidence-Graph-Reranker` unchanged unless updating it intentionally. When copying this GitHub repository to a new Hugging Face Gradio Space, copy `HUGGINGFACE_SPACE_README.md` to the Space's `README.md` to add the SDK metadata. The root GitHub README intentionally stays table-only.
