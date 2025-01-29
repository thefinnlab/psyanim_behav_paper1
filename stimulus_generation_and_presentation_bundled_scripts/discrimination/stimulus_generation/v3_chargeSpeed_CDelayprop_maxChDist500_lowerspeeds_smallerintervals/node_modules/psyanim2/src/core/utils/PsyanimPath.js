class PsyanimPathSegment {

    constructor(p1, p2, distanceToSegment = 0) {

        this._p1 = p1;
        this._p2 = p2;

        this._distanceToSegment = distanceToSegment;

        this._length = p2.clone().subtract(p1).length();
    }

    get distanceToSegment() {
        return this._distanceToSegment;
    }

    get p1() {
        return this._p1;
    }

    get p2() {
        return this._p2;
    }

    get length() {
        return this._length;
    }
}

export default class PsyanimPath {

    constructor(vertices = null) {

        if (vertices)
        {
            if (vertices.length == 1)
            {
                console.error("ERROR: can't create a path from a single vertex!");
            }
            else
            {
                this._segments = this._constructSegmentsFromVertices(vertices);
            }
        }
        else
        {
            this._segments = [];
        }
    }

    get segments() {
        return this._segments;
    }

    _constructSegmentsFromVertices(vertices) {

        let segments = [];

        let distanceToSegment = 0;

        for (let i = 1; i < vertices.length; ++i)
        {
            let segment = new PsyanimPathSegment(
                vertices[i - 1],
                vertices[i],
                distanceToSegment
            );
            
            segments.push(segment);

            distanceToSegment += segment.length;
        }

        return segments;
    }

    getParameter(point) {

        let closestSegment = this.getClosestSegment(point);

        let totalPathLength = this.getTotalLength();

        let parameter = closestSegment.distanceToSegment + this.getParameterForSegment(closestSegment, point);
        parameter /= totalPathLength;

        return parameter;
    }

    getPosition(parameter) {

        // clamp parameter within bounds
        if (parameter > 1.0)
        {
            parameter = 1.0;
        }
        else if (parameter < 0.0)
        {
            parameter = 0.0;
        }

        // find segment parameter is in
        let totalPathLength = this.getTotalLength();
        let parameterDistance = parameter * totalPathLength;

        let parameterSegment = null;

        // TODO: can we simplify this at all?
        if (this._segments.length == 1)
        {
            parameterSegment = this._segments[0];
        }
        else if (parameter == 0.0)
        {
            parameterSegment = this._segments[0];
        }
        else if (parameter == 1.0)
        {
            parameterSegment = this._segments[this._segments.length - 1];
        }
        else
        {
            for (let i = 1; i < this._segments.length; ++i)
            {
                let segment = this._segments[i];

                if (segment.distanceToSegment > parameterDistance)
                {
                    parameterSegment = this._segments[i-1];
                    break;
                }
                else if (segment.distanceToSegment + segment.length > parameterDistance)
                {
                    parameterSegment = this._segments[i];
                    break;
                }
            }    
        }

        let direction = parameterSegment.p2.clone()
            .subtract(parameterSegment.p1)
            .normalize();

        let distanceFromSegmentStart = parameterDistance
            - parameterSegment.distanceToSegment;

        let position = parameterSegment.p1.clone()
            .add(
                direction.scale(distanceFromSegmentStart)
            );

        return position;
    }

    getParameterForSegment(segment, point) {

        let closestPoint = this.getClosestPointToSegment(point, segment);

        return closestPoint.subtract(segment.p1).length();
    }

    getTotalLength() {

        let lastSegment = this._segments[this._segments.length - 1];

        return lastSegment.distanceToSegment + lastSegment.length;
    }

    getClosestPoint(point) {

        let closestSegment = this.getClosestSegment(point);

        return this.getClosestPointToSegment(point, closestSegment);
    }

    getClosestSegment(point) {

        let closestDistance = Infinity;
        let closestSegment = null;

        for (let i = 0; i < this._segments.length; ++i)
        {
            let segment = this._segments[i];

            let distanceToSegment = this.getDistanceToSegment(point, segment);

            // if (distanceToSegment < closestDistance || Math.abs(distanceToSegment - closestDistance) < 1e-3)
            if (distanceToSegment < closestDistance)
            {
                closestDistance = distanceToSegment;
                closestSegment = segment;
            }
        }

        return closestSegment;
    }

    getClosestPointToSegment(point, segment) {

        let p1 = segment.p1;
        let p2 = segment.p2;

        let dv = p2.clone().subtract(p1);

        let norm = dv.dot(dv);

        if (norm < 1e-3)
        {
            // if p1 ~= p2, just return p1
            return p1.clone();
        }

        let pv = point.clone().subtract(p1);

        // t is a projection of point onto the line formed by p1->p2
        let t = pv.dot(dv) / norm;

        if (t < 0)
        {
            return p1.clone();
        }

        if (t > 1)
        {
            return p2.clone();
        }

        return p1.clone().lerp(p2, t);
    }

    getDistanceToSegment(point, segment) {

        let closestPoint = this.getClosestPointToSegment(point, segment);

        return point.clone().subtract(closestPoint).length();
    }
}