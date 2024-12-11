class ConfigBase:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = ConfigBase(**value)
            setattr(self, key, value)
    def get(self, key, default=None):
        return getattr(self, key, default)
    def items(self):
        return self.to_dict().items()
    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigBase):
                result[key] = value.to_dict()
            elif isinstance(value, list):

                result[key] = [
                    v.to_dict() if isinstance(v, ConfigBase) else v for v in value
                ]
            else:
                result[key] = value
        return result 
    def __contains__(self, key):
        return key in self.__dict__
    def __getitem__(self, key):  
        if key in self.__dict__:
            return self.__dict__[key]
        raise KeyError(f"Key '{key}' not found in ConfigBase.")
    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = ConfigBase(**value)
        self.__dict__[key] = value

   

def dict_to_config(config_class, config_dict):
    return config_class(**{
        key: dict_to_config(config_class, value) if isinstance(value, dict) else value
        for key, value in config_dict.items()
    })


def serialize_object(obj):

    if isinstance(obj, dict):
        return {key: serialize_object(value) for key, value in obj.items()}
    elif hasattr(obj, "__dict__"):
        return {key: serialize_object(value) for key, value in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [serialize_object(item) for item in obj]
    else:
        return obj
