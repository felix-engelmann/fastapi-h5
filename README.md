# Fastapi-h5

Fastapi-h5 provides a [fastapi](https://fastapi.tiangolo.com/) [router](https://fastapi.tiangolo.com/reference/apirouter/) that implements the [HSDS](https://github.com/HDFGroup/hsds) (Highly Scalable Data Service) protocol. HSDS is a REST-based service to publish HDF5-like data. The router only requires a `get_data()` function that returns a dictionary (and optionally a reader lock) containing native python scalars, strings, Sequences or NumPy arrays and the relative metadata.&#x20;

## Features

[NeXus](https://manual.nexusformat.org/introduction.html) groups, fields, and attributes are supported, as well as composite datasets to allow you to build a more expressive representation of your data and provide visualization hints for clients.

The published dictionary is reloaded every time a request is received, to enable the publication of live data. The library also offers the possibility to compute the requested dataset on demand, depending on the request parameters, by returning a specialised function as a field.

## Installation

The library can be installed from pypi:

`pip install fastapi-h5`

## Usage

The [example](https://github.com/felix-engelmann/fastapi-h5/tree/main/example) folder contains two IPython notebooks that can be used together as an example server and client. The same folder contains an example app; to access any data published with Fastapi-h5 you can use the [h5pyd](https://github.com/HDFGroup/h5pyd) library, or third-party viewers such as [silx view](https://www.silx.org/doc/silx/latest/applications/view.html) and [h5web](https://github.com/silx-kit/h5web).&#x20;
