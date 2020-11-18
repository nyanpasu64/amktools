from typing import Optional


def offset_add(slice_idx: Optional[int], offset: int):
    if slice_idx is None or slice_idx < 0:
        return slice_idx

    return slice_idx + offset


class StringSlice:
    def __init__(self, string: str, offset: int):
        self.string = string
        self.offset = offset
        if offset > len(string):
            raise IndexError("out of bounds StringSlice constructor")

        self.len = len(string) - offset

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.string[offset_add(item, self.offset)]
        elif isinstance(item, slice):
            return self.string[
                offset_add(item.start or 0, self.offset) : offset_add(
                    item.stop, self.offset
                ) : item.step
            ]
        else:
            raise TypeError(f"unhandled StringSlice index {type(item)}")

    def __len__(self):
        return self.len
