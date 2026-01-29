from setuptools import setup

setup(
    name="rbms_genre_processor",
    version="1.0.1",
    py_modules=["process_genre_fast_reduced_prompt"],
    include_package_data=True,
    package_data={"": ["rbms_terms.json"]},
    install_requires=[
        "pandas",
        "requests",
        "tqdm",
        "openpyxl",
    ],
    python_requires=">=3.8",
)
