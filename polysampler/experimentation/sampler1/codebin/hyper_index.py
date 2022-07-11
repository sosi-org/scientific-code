class MyData:
    def __init__(self):
        pass

    def __getitem__(self, *args, **kargs):
        self.g('__getitem__', *args, **kargs)

    def g(self, name, *args, **kargs):
        print('accessed', name, args, kargs)
        return 'ok'

    def __setitem__(self, *args, **kargs):
        self.g('__setitem__', *args, **kargs)

d = MyData()


# calls `__getitem__` :
# d[{r:5}]
d[2,5,6]
d[{2:5}]  # ({2: 5},) {}
#d[*{2:5}] # error
# d[] error

d[None]  #  (None,) {}
d['ddd']   # ('ddd',) {}
d[None, None]  #  ((None, None),) {}
d[:] # (slice(None, None, None),) {}
d[:,:] # ((slice(None, None, None), slice(None, None, None)),) {}

d[2:4]  #  (slice(2, 4, None),) {}
# print(2:4) # invalid

print('.....')
d[...]  # (Ellipsis,) {}
d[..., 2]  #  ((Ellipsis, 2),) {}
d[1, ...]  # ((1, Ellipsis),) {}


# https://stackoverflow.com/questions/1957780/how-to-override-the-operator-in-python
#d[2:4] = 5
d[2] = 5 #__setitem__:   (2, 5) {}

# https://docs.python.org/3/reference/datamodel.html#object.__getitem__


d[2,3] = 6 # ((2, 3), 6) {}
d[:,3] = 7 # ((slice(None, None, None), 3), 7)


# object.__match_args__


# import email.mime.text

# https://docs.python.org/3/glossary.html#term-qualified-name

PointIndex = namedtuple('PointIndex', ['i'])
#EdgeIndex = namedtuple('EdgeIndex', ['i', 'j'])
Edge = namedtuple('Edge', ['i', 'j'])

# SQL?

