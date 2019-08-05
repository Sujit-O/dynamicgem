#############
Installation
#############

dynamicegem is available in the PyPi's repository.

**Please install** Tensorflow_ **gpu version before installing dynamicgem!** for best performance.

**Prepare your environment**::

    $ sudo apt update
    $ sudo apt install python3-dev python3-pip
    $ sudo pip3 install -U virtualenv

**Create a virtual environment**

If you have tensorflow installed in the root env, do the following::

    $ virtualenv --system-site-packages -p python3 ./venv

If you you want to install tensorflow later, do the following::

    $ virtualenv -p python3 ./venv

Activate the virtual environment using a shell-specific command::

    $ source ./venv/bin/activate

**Upgrade pip**::

    $ pip install --upgrade pip

If you have not installed tensorflow, or not used --system-site-package option while creating venv, install tensorflow first::

    (venv) $ pip install tensorflow

**Install dynamicgem using `pip`**::

    (venv) $ pip install dynamicgem

**Install stable version directly from github repo**::

    (venv) $ git clone https://github.com/Sujit-O/dynamicgem.git
    (venv) $ cd dynamicgem
    (venv) $ python setup.py install

**Install development version directly from github repo**::

    (venv) $ git clone https://github.com/Sujit-O/dynamicgem.git
    (venv) $ cd dynamicgem
    (venv) $ git checkout development
    (venv) $ python setup.py install

.. _GitHub: https://github.com/Sujit-O/dynamicgem/pulls
.. _Tensorflow: https://www.tensorflow.org/install
