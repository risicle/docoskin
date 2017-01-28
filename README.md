# Docoskin

"Onion-skin" visual differences between a reference document image and a scanned copy.

Given an image of a reference document, docoskin will attempt to find key points of the document in a scanned copy,
align the two, auto-correct the image contrasts and display a combined image, featuring sections in red which were
"removed from" the reference document and sections in green which were added to the document.

The intended use for this is comparing a signed, returned scan of a contract with the original version as provided to
the signer to check for unagreed amendments.

Implemented as a python library with a command-line interface. Significantly more power and flexibility is
achievable through accessing the python components.

Depends on opencv and six (and, on python2.7, the `futures` backport).

opencv is always going to be a slightly painful dependency from a python point of view because a) it's a native library
and b) it doesn't really play by python packaging rules supplying an egg. I advocate use of `nix` to solve both these
problems and provide a `default.nix` so that `nix` users can simply perform a

```
$ nix-shell .
```

in the source directory for a development-ready shell (somthing like non-python-specific `virtualenv`).

Those wishing to attempt using the pypi `opencv-python` package can try installing the package with the 'extra'
`pypi_opencv`, which _hypothetically_ should download & install opencv from pypi

```
$ pip install -e .[pypi_opencv]
```

**but** the author has never had any luck with that and has unfavourable opinions of `pip` in general, not to mention the
concept of downloading & installing binaries from pypi.

Preliminary license is GPLv3.
