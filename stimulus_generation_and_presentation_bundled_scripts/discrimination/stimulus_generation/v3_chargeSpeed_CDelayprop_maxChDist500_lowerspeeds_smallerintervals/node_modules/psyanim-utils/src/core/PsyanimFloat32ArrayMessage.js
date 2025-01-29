/**
 *  This is an immutable type - once instantiated, it can no longer be modified!
 */
export default class PsyanimFloat32ArrayMessage {

    constructor(type, userDefinedJsonHeader, payload) {

        this._type = type; // 4 byte integer

        this._userDefinedJsonHeader = userDefinedJsonHeader;

        // encode user defined header into a typed array
        this._userDefinedHeaderString = JSON.stringify(userDefinedJsonHeader);

        this._userDefinedHeaderBytes = new TextEncoder().encode(this._userDefinedHeaderString);
        this._userDefinedHeaderLength = this._userDefinedHeaderBytes.length;

        // handle payload
        this._payload = payload;

        this._payloadBytes = new Uint8Array(this._payload.buffer);
        this._payloadLength = this._payloadBytes.length;

        this._packedMessageLength = PsyanimFloat32ArrayMessage.STANDARD_HEADER_LENGTH + 
            this._userDefinedHeaderLength + this._payloadLength;

        // we pad the message out to an integer-multiple of 4
        this._paddedMessageLength = 4.0 * Math.ceil(this._packedMessageLength / 4.0);

        this._paddingBytesLength = this._paddedMessageLength - this._packedMessageLength;

        this._payloadOffset = PsyanimFloat32ArrayMessage.STANDARD_HEADER_LENGTH + 
            this._userDefinedHeaderLength + this._paddingBytesLength;
    }

    get type() {
        return this._type;
    }

    get userDefinedHeaderLength() {
        return this._userDefinedHeaderLength;
    }

    get paddingBytesLength() {
        return this._paddingBytesLength;
    }

    get payloadLength() {
        return this._payloadLength;
    }

    get userDefinedJsonHeader() {
        return this._userDefinedJsonHeader;
    }

    get payload() {
        return this._payload;
    }

    static fromBytes(buffer) {

        let messageBytes = new Uint8Array(buffer);
        let paddedMessageLength = messageBytes.length;

        if (paddedMessageLength % 4 != 0)
        {
            console.error("ERROR: message length must be an integer multiple of 4!");
        }

        // create a view of the message as 32-bit ints
        let messageViewUint32 = new Uint32Array(messageBytes.buffer, 0, 4);

        let messageType = messageViewUint32[0];
        let userDefinedHeaderLength = messageViewUint32[1];
        let paddingBytesLength = messageViewUint32[2];
        let payloadLength = messageViewUint32[3];

        let userDefinedHeaderString = new TextDecoder().decode(messageBytes.subarray(
            PsyanimFloat32ArrayMessage.STANDARD_HEADER_LENGTH, 
            PsyanimFloat32ArrayMessage.STANDARD_HEADER_LENGTH + userDefinedHeaderLength));

        let userDefinedHeaderJson = JSON.parse(userDefinedHeaderString);

        let payloadOffset = PsyanimFloat32ArrayMessage.STANDARD_HEADER_LENGTH + userDefinedHeaderLength + paddingBytesLength;

        let deserializedPayloadBytes = messageBytes.slice(
            payloadOffset,
            payloadOffset + payloadLength
        );
        
        let deserializedPayload = new Float32Array(deserializedPayloadBytes.buffer);

        let message = new PsyanimFloat32ArrayMessage(messageType, 
            userDefinedHeaderJson, deserializedPayload);

        return message;
    }

    toBytes() {

        /**
         *  Construct full message from the parts
         */

        // allocate empty byte array of proper length
        let message = new Uint8Array(this._paddedMessageLength);

        // create a view of the message as 32-bit ints
        let messageViewUint32 = new Uint32Array(message.buffer, 0, 4);

        // copy the standard header fields into message
        messageViewUint32[0] = this._type;
        messageViewUint32[1] = this._userDefinedHeaderLength;
        messageViewUint32[2] = this._paddingBytesLength;
        messageViewUint32[3] = this._payloadLength;

        // copy the user-defined portion of the header into the message
        message.set(this._userDefinedHeaderBytes, PsyanimFloat32ArrayMessage.STANDARD_HEADER_LENGTH);

        // copy the payload into the message, with necessary padding between header + payload
        message.set(this._payloadBytes, this._payloadOffset);

        return message;
    }
}

/**
 *  Standard header consists of the following 4-byte integer fields, tightly packed:
 * 
 *  - Message Type
 *  - User-defined Header Length
 *  - Padding Bytes Length (inserted at the end of user-defined header, before payload)
 *  - Payload Length
 * 
 *  Note that the total message length should be an integer-multiple of 4 to make
 *  parsing the payload easy using TypedArray APIs.
 */
PsyanimFloat32ArrayMessage.STANDARD_HEADER_LENGTH = 16;