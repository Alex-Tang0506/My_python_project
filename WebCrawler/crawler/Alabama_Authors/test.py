from fake_useragent import UserAgent
if __name__ == '__main__':
    agent__random = UserAgent().random
    print(str(agent__random))