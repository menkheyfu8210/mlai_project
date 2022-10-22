def IOU(p1, p2):
    """Computation of the Intersection Over Union of two bounding boxes.

    Parameters
    ----------
    p1 : list
        List with length 5 holding:

            - float, x coordinate of the top left corner of the bounding box
            - float, y coordinate of the top left corner of the bounding box
            - float, confidence score associated to the bounding box
            - int, width of the bounding box
            - int, heigth of the bounding box

    p2 : list
        List with length 5 holding:

            - float, x coordinate of the top left corner of the bounding box
            - float, y coordinate of the top left corner of the bounding box
            - float, confidence score associated to the bounding box
            - int, width of the bounding box
            - int, heigth of the bounding box

    Returns
    -------
    float, expressing the IOU of the two bounding boxes.
    """
    # x-y coordinates of the detected rectangles
    x1_tl = p1[0]
    x2_tl = p2[0]
    x1_br = p1[0] + p1[3]
    x2_br = p2[0] + p2[3]
    y1_tl = p1[1]
    y2_tl = p2[1]
    y1_br = p1[1] + p1[4]
    y2_br = p2[1] + p2[4]
    # Overlap area
    xOverlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
    yOverlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    overlapArea = xOverlap * yOverlap
    area1 = p1[3] * p1[4]
    area2 = p2[3] * p2[4]
    totalArea = area1 + area2 - overlapArea
    return overlapArea / float(totalArea)

def nms(B, threshold=.5):
    """Non-Maximum Suppression. Given a list B of proposed bounding boxes:

        1.Select from B the proposal with the highest confidence score, remove it
            from B, and add it to an initially empty list D.
        2.Compare the proposal with all other proposals in B by computing the ratio
            between the overlapping area and the total area (IOU).
        3.Remove from B all the proposals with an IOU greater than some threshold.\\
        4.Repeat until B is empty.

    Parameters
    ----------
    B : list
        List of proposed bounding boxes. Each element in the list is a list
        containing:

            - float, x coordinate of the top left corner of the bounding box
            - float, y coordinate of the top left corner of the bounding box
            - float, confidence score associated to the bounding box
            - int, width of the bounding box
            - int, heigth of the bounding box


    threshold : float
        IOU Threshold for the discarding of overlapping bounding boxes.

    Returns
    -------
    list, containing the filtered bounding boxes.
    """
    if len(B) == 0:
        return []
    # Sort the proposals based on confidence score
    B = sorted(B, key=lambda B: B[2], reverse=True)
    D = []
    D.append(B[0])
    B[0] = None
    # Keep only the proposals whose overlapping area with other proposals is 
    # lower than the threshold
    for i, b in enumerate(B):
        if b is not None:
            for d in D:
                if IOU(b, d) > threshold:
                    B[i] = None
                    break
                else:
                    D.append(b)
                    B[i] = None
    return D