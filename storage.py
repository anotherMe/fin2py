class Storage(dict):

    """Dictionary class used to store stock data.

    Provide dictionary storage functionality for our cached stock information

    """
    
    def __getattr__(self, key):
        """Method to get attribute value from dictionary.

        Parameter:
            key: key for value to return

        Returns:
            value stored at the key in the dictionary

        """        
        try:
            return self[key]
        except KeyError:
            return None
        
    def __setattr__(self, key, value):
        """Method to set an attribute in the dictionary.

        Parameter:
            key: key of value to set in the dictionary
            value: value to set in the dictionary using the key

        Returns:
            none
            
        """
        self[key] = value
        
    def __delattr__(self, key):
        """Method to delete an attribute from the dictionary.

        Parameter:
            key: key of value to delete from the dictionary

        Returns:
            none

        """
        try:
            del self[key]
        except KeyError, k:
            raise AttributeError, k

    def __repr__(self):
        """Method to return a string representation of this object.

        Parameter:
            none

        Returns:
            string

        """
        return '<Storage ' + dict.__repr__(self) + '>'
    
    def __getstate__(self):
        """Method to get the entire dictionary.

        Parameter:
            none

        Returns:
            dictionary object
            
        """
        return dict(self)
    
    def __setstate__(self, value):
        """Method to create the entire dictionary.

        Parameter:
            value: dictionary that contains all values to add to our Storage object

        Returns:
            none

        """
        for (k, v) in value.items():
            self[k] = v
