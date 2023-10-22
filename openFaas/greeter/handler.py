def handle(req):
    """handle a request to the function
    Args:
        req (str): request body
    """

    return f"Hello! I've received {len(req)} bytes from you"
