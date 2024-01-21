from torch.utils.data import Dataset



class BaseDataset(Dataset):
    def __init__(self, name):
        super(BaseDataset, self).__init__()
        self.name = name
        print('base dataset')
    
    def __getitem__(self):
        return self.name

    def __len__(self):
        return 1
    
class SonDataset(BaseDataset):
    def __init__(self, name):
        super(SonDataset, self).__init__(name)
        self.name = name
        print('son dataset')





class Father(object):
    def __init__(self, name):
        self.name = name
        print("name: %s" % self.name)
 
    def Get_Name(self):
        return 'Father ' + self.name
 
 
class Son(Father):
    def __init__(self, name):
        super(Son, self).__init__(name)
        print("hi")
        self.name = name
 
    def get_name(self):
        return 'Son '+self.name
    
class GrandSon(Son):
    def __init__(self, name):
        super(GrandSon, self).__init__(name)
        print('grand son')
 
 
if __name__ == '__main__':
    son = SonDataset('I am here')
    # print(son.get_name())