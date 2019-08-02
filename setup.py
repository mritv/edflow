from setuptools import setup, find_packages

setup(
    name="edflow",
    version="0.2",
    description="Logistics for Deep Learning",
    url="https://github.com/pesser/edflow",
    author="Mimo Tilbich et al.",
    author_email="{patrick.esser, johannes.haux}" "@iwr.uni-heidelberg.de",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "pyyaml",
        "opencv-python",
        "tqdm",
        "Pillow",
        "chainer",
        "numpy",
        "scipy",
        "h5py",
        "scikit-learn",
        "scikit-image",
        "natsort",
        "pandas",
        "psutil",
        "pytest",
        "deprecated",
        "fastnumbers",
    ],
    extras_require={"docs": ["sphinx >= 1.4", "sphinx_rtd_theme", "numpy"]},
    zip_safe=False,
    scripts=["edflow/edflow", "edflow/edcache", "edflow/edlist", "edflow/edeval"],
)
