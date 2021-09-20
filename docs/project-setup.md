# Project Setup
This guide will walk through setting up Python and other project dependencies that are required to successfully run this project
<hr>

# Setting Up a Python Development Environment
The first step towards running this project is setting up a Python development environment. This project is written entirely in Python, using pip for dependency management. The *deadsnakes* PPA maintains the latest stable version of Python. For this project, we will require a minimum Python version of 3.8, which we can install via the PPA.

## Installation Procedure for Windows
Python provides official binaries on the official Python site. Download Python by following the instructions below.

1. Go to https://www.python.org/downloads/release/python-386/ to download the Python 3.8.6 binary.
2. On the website, select the `Windows x86 Executable Installer`. This will download the Python executable installer to the Downloads folder
   ![Python for Windows](../assets/images/python-windows-binary.png)
3. Click on the downloaded installer to begin the installation procedure. This will bring up the Python installation guide. 
4. Select "Add Python to Path" as indicated below.
    ![Python Installer](../assets/images/python-installation.png)
5. Click "Install Now" to continue the Python installation and follow all the remaining prompts till the installation process is complete.
6. Close the installer window once the installation is complete.

!!! note "Python commands on windows and Linux"
    Once Python is installed, the procedure for running the code on Linux and Windows is essentially the same.

## Installation Procedure for Linux (Ubuntu)
Let's start by adding the `deadsnakes ppa` to the apt list. To do so, run the following commands:

```shell
$ sudo add-apt-repository ppa:deadsnakes/ppa
$ sudo apt update && sudo apt upgrade
```

!!! note
    You will need a stable internet connection to add the apt repository. If working in a VM behind a firewall, make sure that the proxy server is properly configured to allow access to apt repositories. 

Once we have added the repository and updated the apt sources, we can install Python by running the following command

``` shell
$ sudo apt install python3.8 python3.8-dev python3.8-venv
```
The above command will automatically install PIP which will be used for managing project dependencies. Ensure that the pip version is installed in the environment by running

```shell
$ python3.8 -m ensurepip --default-pip --user
```

!!! note
    The Python binary installed will have the python version attached to it. That is, python{3.x}. You can use a bash alias to map the Python command to the python{3.x} binary. See [this webpage](https://askubuntu.com/questions/320996/how-to-make-python-program-command-execute-python-3) for more information on how to do this.

Test that the Python binary is installed by running

```shell
$ python3.8 --version
# this should return: 3.8.x
```
## Install Anaconda
Anaconda provides various tools and packages for working with Python. MLflow uses conda environments for managing model builds.

Head over to the [Anaconda website](https://repo.anaconda.com/archive/), and in your browser, download the Anaconda installer for Linux-aarch64 (if installing on a Linux machine) or the Windows executable file if installing for Windows. After downloading this installer file to the `~/Downloads` directory, run the following command to install Anaconda

!!! caution
    The file name and location you download to may be different from the one listed below. Make sure to run the right file during this process.

```shell
# on Linux
$ bash ~/Downloads/Anaconda3-2021.04-Linux-aarch64.sh

# on Windows
start /wait "" Anaconda3-2021.04-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%UserProfile%\Anaconda

```
Once you run the command above Anaconda will begin installing `conda`. Accept the user agreement and the default values for all the prompts provided. 

When the installation is complete and successful, run the following command to activate the `conda` environment:
```shell
$ conda init
```
## Clone the Repository
To clone the repository, run
```shell
$ git clone https://github.com/inteligenciaambientalsma/ComplaintModelSMA.git
```
This will clone the repository into the `ComplaintModelSMA` folder which will serve as the project root.

!!! note "Private Repository"
    At the time of writing, the link above exists as a private repository. You may need to request access from the SMA
    team in order to clone the repository. A public facing repository will be made available by the DSSGx team soon.

## Installing Project Dependencies
Now we can go ahead and install the project dependencies. To do so, run the following command in the project root:

```shell
$ cd ComplaintModelSMA/
$ pip install -r requirements.txt
```
This will install all the dependencies required to run this project. It will also install the dependencies required to build and run the documentation. After installing the dependencies, we can proceed to the data preparation and model training steps. See the next section of this documentation which provides details of the data preparation. See [Data Preparation](../set-up/preparing-the-data).

## Building the Project Documentation
This documentation is built using [MKDocs](https://www.mkdocs.org/) using the [`mkdocs-gitbook`](https://lramage.gitlab.io/mkdocs-gitbook-theme/) theme. This documentation also depends on the [`mkdocstrings`](https://github.com/mkdocstrings/mkdocstrings) plugin in order to dynamically import code snippets directly from files.

**Install MKDocs and documentation plugins**
```shell
$ pip install mkdocs mkdocs-gitbook mkdocstrings
```

**Preview the documentation**
```shell
$ mkdocs serve
```
This will start the MKDocs development server on port 8000 (the default mkdocs port).
!!! note "Serving on a different port"
    You can serve the documentation on a different port by running
    `mkdocs serve --dev-addr localhost:<PORT>`

**Deploy the documentation to GitHub pages**
```shell
   $ mkdocs gh-deploy
```
   This command builds the documentation in the background, generating static files and placing them in the `gh-pages` branch of the repository. To learn more about the deployment process, see [Deploying Your Docs](https://www.mkdocs.org/user-guide/deploying-your-docs/).

# Conclusion
Now that we have installed Python, pip and conda, we can begin running the project. In the next section of the documentation we will describe in detail the data preparing steps required for training the machine learning models.
