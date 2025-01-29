import {
    PsyanimFirebaseClientBase
} from 'psyanim-utils';

// firebase imports
import firebase from "firebase/compat/app";
import "firebase/compat/firestore";

export default class PsyanimFirebaseBrowserClient extends PsyanimFirebaseClientBase {

    constructor(firebaseConfig) {
        
        super(firebaseConfig);
    }

    _initFirebase(firebaseConfig) {

        firebase.initializeApp(firebaseConfig);
    }

    _initFirestore() {

        return firebase.firestore();
    }

    _getServerTimestamp() {

        return firebase.firestore.FieldValue.serverTimestamp();
    }
}